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
from typing import Sequence, Optional
import math
import logging

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import rng_choice
from vkit.element import LexiconCollection
from vkit.engine.interface import (
    NoneTypeEngineInitConfig,
    Engine,
    EngineExecutorFactory,
    EngineExecutorAggregator,
)
from vkit.engine.font.type import (
    FontCollection,
    FontVariant,
    FontEngineRunConfigGlyphSequence,
)
from vkit.engine.char_sampler.type import CharSamplerEngineRunConfig

logger = logging.getLogger(__name__)


@attrs.define
class CharAndFontSamplerEngineRunConfig:
    height: int
    width: int
    glyph_sequence: FontEngineRunConfigGlyphSequence = \
        FontEngineRunConfigGlyphSequence.HORI_DEFAULT
    num_chars_factor: float = 1.1
    num_chars: Optional[int] = None


@attrs.define
class CharAndFontSamplerEngineInitResource:
    lexicon_collection: LexiconCollection
    font_collection: FontCollection
    char_sampler_engine_executor_aggregator: EngineExecutorAggregator[
        CharSamplerEngineRunConfig,
        Sequence[str],
    ]  # yapf: disable


@attrs.define
class CharAndFont:
    chars: Sequence[str]
    font_variant: FontVariant


class CharAndFontSamplerEngine(
    Engine[
        NoneTypeEngineInitConfig,
        CharAndFontSamplerEngineInitResource,
        CharAndFontSamplerEngineRunConfig,
        Optional[CharAndFont],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'default'

    def __init__(
        self,
        init_config: NoneTypeEngineInitConfig,
        init_resource: Optional[CharAndFontSamplerEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        assert init_resource
        self.font_collection = init_resource.font_collection
        self.lexicon_collection = init_resource.lexicon_collection
        self.char_sampler_engine_executor_aggregator = init_resource.char_sampler_engine_executor_aggregator

    @classmethod
    def estimate_num_chars(cls, run_config: CharAndFontSamplerEngineRunConfig):
        if run_config.num_chars:
            return run_config.num_chars

        if run_config.glyph_sequence == FontEngineRunConfigGlyphSequence.HORI_DEFAULT:
            num_chars = run_config.width / run_config.height
        elif run_config.glyph_sequence == FontEngineRunConfigGlyphSequence.VERT_DEFAULT:
            num_chars = run_config.height / run_config.width
        else:
            raise NotImplementedError()

        num_chars *= run_config.num_chars_factor
        return math.ceil(num_chars)

    def run(
        self,
        run_config: CharAndFontSamplerEngineRunConfig,
        rng: RandomGenerator,
    ) -> Optional[CharAndFont]:
        # Sample chars.
        num_chars = self.estimate_num_chars(run_config)
        chars = self.char_sampler_engine_executor_aggregator.run(
            CharSamplerEngineRunConfig(
                num_chars=num_chars,
                enable_aggregator_mode=True,
            ),
            rng,
        )
        logger.debug(f'chars={chars}')

        # Sample font variant.
        font_metas = self.font_collection.filter_font_metas(chars)
        if not font_metas:
            logger.warning(f'Cannot sample font_metas for chars={chars}')
            return None

        font_meta = rng_choice(rng, font_metas)
        variant_idx = int(rng.integers(0, font_meta.num_font_variants))
        font_variant = font_meta.get_font_variant(variant_idx)

        return CharAndFont(chars=chars, font_variant=font_variant)


char_and_font_sampler_engine_executor_factory = EngineExecutorFactory(CharAndFontSamplerEngine)
