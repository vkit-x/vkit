from typing import Sequence, Mapping, Any, Union
from enum import Enum, unique

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import (
    normalize_to_keys_and_probs,
    rng_choice,
    PathType,
)
from vkit.element import Image
from vkit.engine.image import image_factory
from .page_shape import PageShapeStep
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)


@attrs.define
class PageBackgroundStepConfig:
    image_configs: Union[Sequence[Mapping[str, Any]], PathType]
    weight_image: float = 0.8
    weight_random_grayscale: float = 0.2
    grayscale_min: int = 127
    grayscale_max: int = 255


@attrs.define
class PageBackgroundStepOutput:
    background_image: Image


@unique
class PageBackgroundStepKey(Enum):
    IMAGE = 'image'
    RANDOM_GRAYSCALE = 'random_grayscale'


class PageBackgroundStep(
    PipelineStep[
        PageBackgroundStepConfig,
        PageBackgroundStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageBackgroundStepConfig):
        super().__init__(config)

        self.image_aggregator = image_factory.create(self.config.image_configs)

        self.keys, self.probs = normalize_to_keys_and_probs([
            (
                PageBackgroundStepKey.IMAGE,
                self.config.weight_image,
            ),
            (
                PageBackgroundStepKey.RANDOM_GRAYSCALE,
                self.config.weight_random_grayscale,
            ),
        ])

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_shape_step_output = state.get_pipeline_step_output(PageShapeStep)
        height = page_shape_step_output.height
        width = page_shape_step_output.width

        key = rng_choice(rng, self.keys, probs=self.probs)
        if key == PageBackgroundStepKey.IMAGE:
            background_image = self.image_aggregator.run(
                {
                    'height': height,
                    'width': width,
                },
                rng,
            )

        elif key == PageBackgroundStepKey.RANDOM_GRAYSCALE:
            grayscale_value = rng.integers(self.config.grayscale_min, self.config.grayscale_max + 1)
            background_image = Image.from_shape(
                (height, width),
                num_channels=3,
                value=grayscale_value,
            )

        else:
            raise NotImplementedError()

        return PageBackgroundStepOutput(background_image=background_image)


page_background_step_factory = PipelineStepFactory(PageBackgroundStep)
