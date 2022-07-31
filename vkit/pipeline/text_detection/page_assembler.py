import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import Shapable, Box, Image
from vkit.engine.seal_impression import fill_text_line_to_seal_impression
from vkit.engine.distortion import rotate
from ..interface import (
    PipelineStep,
    NoneTypePipelineStepConfig,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep
from .page_background import PageBackgroundStep
from .page_image import PageImageStep, PageImageCollection
from .page_qrcode import PageQrcodeStep
from .page_barcode import PageBarcodeStep
from .page_text_line import (
    PageTextLineStep,
    PageTextLineCollection,
    PageSealImpressionTextLineCollection,
)
from .page_text_line_label import (
    PageTextLineLabelStep,
    PageCharPolygonCollection,
    PageTextLinePolygonCollection,
)
from .page_text_line_bounding_box import PageTextLineBoundingBoxStep


@attrs.define
class Page(Shapable):
    image: Image
    page_image_collection: PageImageCollection
    page_text_line_collection: PageTextLineCollection
    page_seal_impression_text_line_collection: PageSealImpressionTextLineCollection
    page_char_polygon_collection: PageCharPolygonCollection
    page_text_line_polygon_collection: PageTextLinePolygonCollection

    @property
    def height(self):
        return self.image.height

    @property
    def width(self):
        return self.image.width


@attrs.define
class PageAssemblerStepOutput:
    page: Page


class PageAssemblerStep(
    PipelineStep[
        NoneTypePipelineStepConfig,
        PageAssemblerStepOutput,
    ]
):  # yapf: disable

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_layout_step_output = state.get_pipeline_step_output(PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        page_background_step_output = state.get_pipeline_step_output(PageBackgroundStep)
        background_image = page_background_step_output.background_image

        page_image_step_output = state.get_pipeline_step_output(PageImageStep)
        page_image_collection = page_image_step_output.page_image_collection

        page_qrcode_step_output = state.get_pipeline_step_output(PageQrcodeStep)
        page_barcode_step_output = state.get_pipeline_step_output(PageBarcodeStep)

        page_text_line_step_output = state.get_pipeline_step_output(PageTextLineStep)
        page_text_line_collection = page_text_line_step_output.page_text_line_collection
        page_seal_impression_text_line_collection = \
            page_text_line_step_output.page_seal_impression_text_line_collection

        page_text_line_bounding_box_step_output = \
            state.get_pipeline_step_output(PageTextLineBoundingBoxStep)
        text_line_bounding_box_score_maps = page_text_line_bounding_box_step_output.score_maps
        text_line_bounding_box_colors = page_text_line_bounding_box_step_output.colors

        page_text_line_label_step_output = state.get_pipeline_step_output(PageTextLineLabelStep)
        page_char_polygon_collection = \
            page_text_line_label_step_output.page_char_polygon_collection
        page_text_line_polygon_collection = \
            page_text_line_label_step_output.page_text_line_polygon_collection

        # Page background.
        assert background_image.mat.shape == (page_layout.height, page_layout.width, 3)
        assembled_image = background_image.copy()

        # Page images.
        for page_image in page_image_collection.page_images:
            page_image.box.fill_image(
                assembled_image,
                page_image.image,
                alpha=page_image.alpha,
            )

        # Page QR codes.
        for qrcode_score_map in page_qrcode_step_output.qrcode_score_maps:
            assembled_image[qrcode_score_map] = (0, 0, 0)
        # Page Bar codes.
        for barcode_score_map in page_barcode_step_output.barcode_score_maps:
            assembled_image[barcode_score_map] = (0, 0, 0)

        # Page text line bounding boxes.
        for text_line_bounding_box_score_map, text_line_bounding_box_color in zip(
            text_line_bounding_box_score_maps, text_line_bounding_box_colors
        ):
            assembled_image[text_line_bounding_box_score_map] = text_line_bounding_box_color

        # Page text lines.
        for text_line in page_text_line_collection.text_lines:
            if text_line.score_map:
                text_line.score_map.fill_image(assembled_image, text_line.glyph_color)
            else:
                text_line.mask.fill_image(assembled_image, text_line.image)

        # Page seal impressions.
        for seal_impression, seal_impression_resource in zip(
            page_seal_impression_text_line_collection.seal_impressions,
            page_seal_impression_text_line_collection.seal_impression_resources,
        ):
            alpha = seal_impression.alpha
            color = seal_impression.color

            # Prepare foreground (text) and background.
            background_mask = seal_impression.background_mask
            text_line_filled_score_map = fill_text_line_to_seal_impression(
                seal_impression,
                seal_impression_resource.text_line_slot_indices,
                seal_impression_resource.text_lines,
                seal_impression_resource.internal_text_line,
            )

            # Rotate, shift, and trim.
            rotated_result = rotate.distort(
                {'angle': seal_impression_resource.angle},
                mask=background_mask,
                score_map=text_line_filled_score_map,
            )
            assert rotated_result.mask
            background_mask = rotated_result.mask
            assert rotated_result.score_map
            text_line_filled_score_map = rotated_result.score_map
            assert background_mask.shape == text_line_filled_score_map.shape

            box_center_point = seal_impression_resource.box.get_center_point()
            up = box_center_point.y - background_mask.height // 2
            down = up + background_mask.height - 1
            left = box_center_point.x - background_mask.width // 2
            right = left + background_mask.width - 1

            if up < 0 or down >= assembled_image.height \
                    or left < 0 or right >= assembled_image.width:
                extract_up = 0
                if up < 0:
                    extract_up = abs(up)
                    up = 0

                extract_down = background_mask.height - 1
                if down >= assembled_image.height:
                    extract_down = background_mask.height - 1 - (assembled_image.height + 1 - down)
                    down = assembled_image.height - 1

                extract_left = 0
                if left < 0:
                    extract_left = abs(left)
                    left = 0

                extract_right = background_mask.width - 1
                if right >= assembled_image.width:
                    extract_right = background_mask.width - 1 - (assembled_image.width + 1 - right)
                    right = assembled_image.width - 1

                extract_box = Box(
                    up=extract_up,
                    down=extract_down,
                    left=extract_left,
                    right=extract_right,
                )
                background_mask = extract_box.extract_mask(background_mask)
                text_line_filled_score_map = extract_box.extract_score_map(
                    text_line_filled_score_map
                )

            # Rendering.
            box = Box(up=up, down=down, left=left, right=right)
            background_mask.to_box_attached(box).fill_image(
                assembled_image,
                color,
                alpha=alpha,
            )
            text_line_filled_score_map.to_box_attached(box).fill_image(
                assembled_image,
                color,
            )

        page = Page(
            image=assembled_image,
            page_image_collection=page_image_collection,
            page_text_line_collection=page_text_line_collection,
            page_seal_impression_text_line_collection=page_seal_impression_text_line_collection,
            page_char_polygon_collection=page_char_polygon_collection,
            page_text_line_polygon_collection=page_text_line_polygon_collection,
        )
        return PageAssemblerStepOutput(page=page)


page_assembler_step_factory = PipelineStepFactory(PageAssemblerStep)
