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
from typing import Sequence, Mapping, Any, List, Union

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import PathType, rng_choice
from vkit.element import Image, Box
from vkit.engine.image import image_engine_executor_aggregator_factory
from vkit.mechanism.distortion import rotate
from ..interface import PipelineStep, PipelineStepFactory
from .page_layout import PageLayoutStepOutput


@attrs.define
class PageImageStepConfig:
    image_configs: Union[Sequence[Mapping[str, Any]], PathType]
    alpha_min: float = 0.25
    alpha_max: float = 1.0


@attrs.define
class PageImageStepInput:
    page_layout_step_output: PageLayoutStepOutput


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
    # For filling inactive region caused by distortion.
    page_bottom_layer_image: Image


class PageImageStep(
    PipelineStep[
        PageImageStepConfig,
        PageImageStepInput,
        PageImageStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageImageStepConfig):
        super().__init__(config)

        self.image_engine_executor_aggregator = \
            image_engine_executor_aggregator_factory.create(self.config.image_configs)

    def run(self, input: PageImageStepInput, rng: RandomGenerator):
        page_layout_step_output = input.page_layout_step_output
        page_layout = page_layout_step_output.page_layout

        page_images: List[PageImage] = []
        for layout_image in page_layout.layout_images:
            image = self.image_engine_executor_aggregator.run(
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

        page_bottom_layer_image = self.image_engine_executor_aggregator.run(
            {
                'height': 0,
                'width': 0,
                'disable_resizing': True,
            },
            rng,
        )
        # Random rotate.
        rotate_angle = rng_choice(rng, (0, 90, 180, 270))
        page_bottom_layer_image = rotate.distort_image(
            {'angle': rotate_angle},
            page_bottom_layer_image,
        )

        return PageImageStepOutput(
            page_image_collection=page_image_collection,
            page_bottom_layer_image=page_bottom_layer_image,
        )


page_image_step_factory = PipelineStepFactory(PageImageStep)
