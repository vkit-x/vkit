import attrs
from numpy.random import RandomState

from vkit.element import Image, Shapable
from ..interface import (
    PipelineStep,
    NoneTypePipelineStepConfig,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep
from .page_background import PageBackgroundStep
from .page_image import PageImageStep, PageImageCollection
from .page_text_line import PageTextLineStep, PageTextLineCollection
from .page_text_line_label import PageTextLineLabelStep, PageTextLinePolygonCollection


@attrs.define
class Page(Shapable):
    image: Image
    page_image_collection: PageImageCollection
    page_text_line_collection: PageTextLineCollection
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

    def run(self, state: PipelineState, rnd: RandomState):
        page_layout_step_output = self.get_output(state, PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        page_background_step_output = self.get_output(state, PageBackgroundStep)
        background_image = page_background_step_output.background_image

        page_image_step_output = self.get_output(state, PageImageStep)
        page_image_collection = page_image_step_output.page_image_collection

        page_text_line_step_output = self.get_output(state, PageTextLineStep)
        page_text_line_collection = page_text_line_step_output.page_text_line_collection

        page_text_line_label_step_output = self.get_output(state, PageTextLineLabelStep)
        page_text_line_polygon_collection = \
            page_text_line_label_step_output.page_text_line_polygon_collection

        # Page background.
        assert background_image.mat.shape == (page_layout.height, page_layout.width, 3)
        assembled_image = background_image.copy()

        # Page images.
        for page_image in page_image_collection.page_images:
            page_image.box.fill_image(assembled_image, page_image.image)

        # Page text lines.
        for text_line in page_text_line_collection.text_lines:
            # Fill only the masked pixels.
            if text_line.score_map:
                text_line.score_map.fill_image(
                    assembled_image,
                    text_line.glyph_color,
                )
            else:
                text_line.mask.fill_image(
                    assembled_image,
                    text_line.image,
                )

        page = Page(
            image=assembled_image,
            page_image_collection=page_image_collection,
            page_text_line_collection=page_text_line_collection,
            page_text_line_polygon_collection=page_text_line_polygon_collection,
        )
        return PageAssemblerStepOutput(page=page)


page_assembler_step_factory = PipelineStepFactory(PageAssemblerStep)
