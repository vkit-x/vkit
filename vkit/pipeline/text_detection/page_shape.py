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
import math

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import rng_choice
from ..interface import PipelineStep, PipelineStepFactory


@attrs.define
class PageShapeStepConfig:
    aspect_ratios: Sequence[float] = attrs.field(factory=lambda: (1 / 1.4142, 1.4142))
    # NOTE: to ensure the minimum font size >= 18 pixels.
    area: int = 2522**2


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
