# Copyright 2022 vkit-x Administrator. All Rights Reserved.
#
# This project (vkit-x/vkit) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
from typing import Sequence

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import Shapable, Box, Image
from vkit.engine.seal_impression import fill_text_line_to_seal_impression
from vkit.mechanism.distortion import rotate
from ..interface import PipelineStep, PipelineStepFactory
from .page_layout import (
    PageLayoutStepOutput,
    DisconnectedTextRegion,
    NonTextRegion,
)
from .page_background import PageBackgroundStepOutput
from .page_image import PageImageStepOutput, PageImageCollection
from .page_barcode import PageBarcodeStepOutput
from .page_text_line import (
    PageTextLineStepOutput,
    PageTextLineCollection,
    PageSealImpressionTextLineCollection,
)
from .page_non_text_symbol import PageNonTextSymbolStepOutput
from .page_text_line_label import (
    PageTextLineLabelStepOutput,
    PageCharPolygonCollection,
    PageTextLinePolygonCollection,
)
from .page_text_line_bounding_box import PageTextLineBoundingBoxStepOutput


@attrs.define
class PageAssemblerStepConfig:
    pass


@attrs.define
class PageAssemblerStepInput:
    page_layout_step_output: PageLayoutStepOutput
    page_background_step_output: PageBackgroundStepOutput
    page_image_step_output: PageImageStepOutput
    page_barcode_step_output: PageBarcodeStepOutput
    page_text_line_step_output: PageTextLineStepOutput
    page_non_text_symbol_step_output: PageNonTextSymbolStepOutput
    page_text_line_bounding_box_step_output: PageTextLineBoundingBoxStepOutput
    page_text_line_label_step_output: PageTextLineLabelStepOutput


@attrs.define
class PageDisconnectedTextRegionCollection:
    disconnected_text_regions: Sequence[DisconnectedTextRegion]

    def to_polygons(self):
        for disconnected_text_region in self.disconnected_text_regions:
            yield disconnected_text_region.polygon


@attrs.define
class PageNonTextRegionCollection:
    non_text_regions: Sequence[NonTextRegion]

    def to_polygons(self):
        for non_text_region in self.non_text_regions:
            yield non_text_region.polygon


@attrs.define
class Page(Shapable):
    image: Image
    page_image_collection: PageImageCollection
    page_bottom_layer_image: Image
    page_text_line_collection: PageTextLineCollection
    page_seal_impression_text_line_collection: PageSealImpressionTextLineCollection
    page_char_polygon_collection: PageCharPolygonCollection
    page_text_line_polygon_collection: PageTextLinePolygonCollection
    page_disconnected_text_region_collection: PageDisconnectedTextRegionCollection
    page_non_text_region_collection: PageNonTextRegionCollection

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
        PageAssemblerStepConfig,
        PageAssemblerStepInput,
        PageAssemblerStepOutput,
    ]
):  # yapf: disable

    def run(self, input: PageAssemblerStepInput, rng: RandomGenerator):
        page_layout_step_output = input.page_layout_step_output
        page_layout = page_layout_step_output.page_layout

        page_background_step_output = input.page_background_step_output
        background_image = page_background_step_output.background_image

        page_image_step_output = input.page_image_step_output
        page_image_collection = page_image_step_output.page_image_collection
        page_bottom_layer_image = page_image_step_output.page_bottom_layer_image

        page_barcode_step_output = input.page_barcode_step_output

        page_text_line_step_output = input.page_text_line_step_output
        page_text_line_collection = page_text_line_step_output.page_text_line_collection
        page_seal_impression_text_line_collection = \
            page_text_line_step_output.page_seal_impression_text_line_collection

        page_non_text_symbol_step_output = input.page_non_text_symbol_step_output

        page_text_line_bounding_box_step_output = input.page_text_line_bounding_box_step_output
        text_line_bounding_box_score_maps = page_text_line_bounding_box_step_output.score_maps
        text_line_bounding_box_colors = page_text_line_bounding_box_step_output.colors

        page_text_line_label_step_output = input.page_text_line_label_step_output
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

        # Page barcodes.
        for barcode_qr_score_map in page_barcode_step_output.barcode_qr_score_maps:
            assembled_image[barcode_qr_score_map] = (0, 0, 0)
        for barcode_code39_score_map in page_barcode_step_output.barcode_code39_score_maps:
            assembled_image[barcode_code39_score_map] = (0, 0, 0)

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

        # Page non-text symbols.
        for image, box, alpha in zip(
            page_non_text_symbol_step_output.images,
            page_non_text_symbol_step_output.boxes,
            page_non_text_symbol_step_output.alphas,
        ):
            box.fill_image(assembled_image, value=image, alpha=alpha)

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
                    extract_down -= down + 1 - assembled_image.height
                    down = assembled_image.height - 1

                extract_left = 0
                if left < 0:
                    extract_left = abs(left)
                    left = 0

                extract_right = background_mask.width - 1
                if right >= assembled_image.width:
                    extract_right -= right + 1 - assembled_image.width
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
            box.fill_image(assembled_image, value=color, image_mask=background_mask, alpha=alpha)
            box.fill_image(assembled_image, value=color, alpha=text_line_filled_score_map)

        # For char-level polygon regression.
        page_disconnected_text_region_collection = PageDisconnectedTextRegionCollection(
            page_layout.disconnected_text_regions
        )

        # For sampling negative text region area.
        page_non_text_region_collection = PageNonTextRegionCollection(page_layout.non_text_regions)

        page = Page(
            image=assembled_image,
            page_image_collection=page_image_collection,
            page_bottom_layer_image=page_bottom_layer_image,
            page_text_line_collection=page_text_line_collection,
            page_seal_impression_text_line_collection=page_seal_impression_text_line_collection,
            page_char_polygon_collection=page_char_polygon_collection,
            page_text_line_polygon_collection=page_text_line_polygon_collection,
            page_disconnected_text_region_collection=page_disconnected_text_region_collection,
            page_non_text_region_collection=page_non_text_region_collection,
        )
        return PageAssemblerStepOutput(page=page)


page_assembler_step_factory = PipelineStepFactory(PageAssemblerStep)
