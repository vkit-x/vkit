from typing import Sequence, Dict, Any, Union
from enum import Enum, unique

import attrs
from numpy.random import RandomState

from vkit.utility import (
    normalize_to_keys_and_probs,
    rnd_choice,
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
    image_configs: Union[Sequence[Dict[str, Any]], PathType]
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

    def run(self, state: PipelineState, rnd: RandomState):
        page_shape_step_output = self.get_output(state, PageShapeStep)
        height = page_shape_step_output.height
        width = page_shape_step_output.width

        key = rnd_choice(rnd, self.keys, probs=self.probs)
        if key == PageBackgroundStepKey.IMAGE:
            background_image = self.image_aggregator.run(
                {
                    'height': height,
                    'width': width,
                },
                rnd,
            )

        elif key == PageBackgroundStepKey.RANDOM_GRAYSCALE:
            grayscale_value = rnd.randint(self.config.grayscale_min, self.config.grayscale_max + 1)
            background_image = Image.from_shape(
                (height, width),
                num_channels=3,
                value=grayscale_value,
            )

        else:
            raise NotImplementedError()

        return PageBackgroundStepOutput(background_image=background_image)


page_background_step_factory = PipelineStepFactory(PageBackgroundStep)
