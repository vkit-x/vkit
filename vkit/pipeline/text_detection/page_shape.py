from typing import Sequence
import math

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import rng_choice
from ..interface import PipelineStep, PipelineStepFactory


@attrs.define
class PageShapeStepConfig:
    aspect_ratios: Sequence[float] = attrs.field(factory=lambda: (1 / 1.4142, 1.4142))
    # NOTE: to ensure the minimum font size >= 12 pixels.
    area: int = 1681**2


@attrs.define
class PageShapeStepInput:
    pass


@attrs.define
class PageShapeStepOutput:
    height: int
    width: int


class PageShapeStep(
    PipelineStep[
        PageShapeStepConfig,
        PageShapeStepInput,
        PageShapeStepOutput,
    ]
):  # yapf: disable

    def run(self, input: PageShapeStepInput, rng: RandomGenerator):
        aspect_ratio = rng_choice(rng, self.config.aspect_ratios)
        height = round(math.sqrt(self.config.area / aspect_ratio))
        width = round(aspect_ratio * height)
        assert height > 0 and width > 0

        return PageShapeStepOutput(height=height, width=width)


page_shape_step_factory = PipelineStepFactory(PageShapeStep)
