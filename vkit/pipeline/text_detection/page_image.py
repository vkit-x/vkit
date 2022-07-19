from typing import Sequence, Mapping, Any, List, Union

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import PathType
from vkit.element import Image, Box
from vkit.engine.image import image_factory
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep


@attrs.define
class PageImageStepConfig:
    image_configs: Union[Sequence[Mapping[str, Any]], PathType]
    alpha_min: float = 0.25
    alpha_max: float = 1.0


@attrs.define
class PageImage:
    image: Image
    box: Box
    alpha: float


@attrs.define
class PageImageCollection:
    height: int
    width: int
    page_images: Sequence[PageImage]


@attrs.define
class PageImageStepOutput:
    page_image_collection: PageImageCollection


class PageImageStep(
    PipelineStep[
        PageImageStepConfig,
        PageImageStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageImageStepConfig):
        super().__init__(config)

        self.image_aggregator = image_factory.create(self.config.image_configs)

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_layout_step_output = state.get_pipeline_step_output(PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        page_images: List[PageImage] = []
        for layout_image in page_layout.layout_images:
            image = self.image_aggregator.run(
                {
                    'height': layout_image.box.height,
                    'width': layout_image.box.width,
                },
                rng,
            )
            alpha = float(rng.uniform(self.config.alpha_min, self.config.alpha_max))
            page_images.append(PageImage(
                image=image,
                box=layout_image.box,
                alpha=alpha,
            ))

        page_image_collection = PageImageCollection(
            height=page_layout.height,
            width=page_layout.width,
            page_images=page_images,
        )
        return PageImageStepOutput(page_image_collection=page_image_collection)


page_image_step_factory = PipelineStepFactory(PageImageStep)
