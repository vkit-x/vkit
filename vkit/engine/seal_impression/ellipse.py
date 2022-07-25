from typing import Sequence, List, Optional

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.engine.interface import Engine
from .type import (
    SealImpressionEngineResource,
    SealImpressionEngineRunConfig,
    SealImpressionLayout,
)


@attrs.define
class EllipseSealImpressionEngineConfig:
    pass


class EllipseSealImpressionEngine(
    Engine[
        EllipseSealImpressionEngineConfig,
        SealImpressionEngineResource,
        SealImpressionEngineRunConfig,
        Sequence[SealImpressionLayout],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'ellipse'

    def __init__(
        self,
        config: EllipseSealImpressionEngineConfig,
        resource: Optional[SealImpressionEngineResource] = None
    ):
        super().__init__(config, resource)

    def run(self, config: SealImpressionEngineRunConfig, rng: RandomGenerator):
        pass
