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
from typing import Sequence, List, Union, Mapping, Any

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import PathType
from vkit.element import Box
from vkit.engine.seal_impression import (
    seal_impression_engine_executor_aggregator_factory,
    SealImpression,
)
from ..interface import PipelineStep, PipelineStepFactory
from .page_layout import PageLayoutStepOutput


@attrs.define
class PageSealImpresssionStepConfig:
    seal_impression_configs: Union[Sequence[Mapping[str, Any]], PathType]


@attrs.define
class PageSealImpresssionStepInput:
    page_layout_step_output: PageLayoutStepOutput


@attrs.define
class PageSealImpresssionStepOutput:
    seal_impressions: Sequence[SealImpression]
    boxes: Sequence[Box]
    angles: Sequence[int]


class PageSealImpresssionStep(
    PipelineStep[
        PageSealImpresssionStepConfig,
        PageSealImpresssionStepInput,
        PageSealImpresssionStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageSealImpresssionStepConfig):
        super().__init__(config)

        self.seal_impression_engine_executor_aggregator = \
            seal_impression_engine_executor_aggregator_factory.create(
                self.config.seal_impression_configs
            )

    def run(self, input: PageSealImpresssionStepInput, rng: RandomGenerator):
        page_layout_step_output = input.page_layout_step_output
        page_layout = page_layout_step_output.page_layout

        seal_impressions: List[SealImpression] = []
        boxes: List[Box] = []
        angles: List[int] = []
        for layout_seal_impression in page_layout.layout_seal_impressions:
            box = layout_seal_impression.box
            seal_impressions.append(
                self.seal_impression_engine_executor_aggregator.run(
                    {
                        'height': box.height,
                        'width': box.width
                    },
                    rng,
                )
            )
            boxes.append(box)
            angles.append(layout_seal_impression.angle)

        return PageSealImpresssionStepOutput(
            seal_impressions=seal_impressions,
            boxes=boxes,
            angles=angles,
        )


page_seal_impresssion_step_factory = PipelineStepFactory(PageSealImpresssionStep)
