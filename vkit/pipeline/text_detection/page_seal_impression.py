from typing import Sequence, List, Union, Mapping, Any

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import PathType
from vkit.element import Box
from vkit.engine.seal_impression import SealImpression, seal_impression_factory
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep


@attrs.define
class PageSealImpresssionStepConfig:
    seal_impression_configs: Union[Sequence[Mapping[str, Any]], PathType]


@attrs.define
class PageSealImpresssionStepOutput:
    seal_impressions: Sequence[SealImpression]
    boxes: Sequence[Box]
    angles: Sequence[int]


class PageSealImpresssionStep(
    PipelineStep[
        PageSealImpresssionStepConfig,
        PageSealImpresssionStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageSealImpresssionStepConfig):
        super().__init__(config)

        self.seal_impression_aggregator = seal_impression_factory.create(
            self.config.seal_impression_configs
        )

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_layout_step_output = state.get_pipeline_step_output(PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        seal_impressions: List[SealImpression] = []
        boxes: List[Box] = []
        angles: List[int] = []
        for layout_seal_impression in page_layout.layout_seal_impressions:
            box = layout_seal_impression.box
            seal_impressions.append(
                self.seal_impression_aggregator.run(
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
