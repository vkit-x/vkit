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
import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import Mask
from ..interface import (
    Engine,
    EngineExecutorFactory,
    NoneTypeEngineInitResource,
)
from .type import CharMaskEngineRunConfig, CharMask


@attrs.define
class CharMaskDefaultEngineInitConfig:
    pass


class CharMaskDefaultEngine(
    Engine[
        CharMaskDefaultEngineInitConfig,
        NoneTypeEngineInitResource,
        CharMaskEngineRunConfig,
        CharMask,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'default'

    def run(self, run_config: CharMaskEngineRunConfig, rng: RandomGenerator) -> CharMask:
        mask = Mask.from_shape((run_config.height, run_config.width))
        for polygon in run_config.char_polygons:
            polygon.fill_mask(mask)
        return CharMask(mask=mask)


char_mask_default_engine_executor_factory = EngineExecutorFactory(CharMaskDefaultEngine)
