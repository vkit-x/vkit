from typing import Sequence, Optional
import math
import logging

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import rng_choice
from vkit.element import LexiconCollection
from vkit.engine.interface import (
    NoneTypeEngineConfig,
    Engine,
    EngineFactory,
    EngineRunnerAggregator,
)
from vkit.engine.font.type import (
    FontCollection,
    FontVariant,
    FontEngineRunConfigGlyphSequence,
)
from vkit.engine.char_sampler.type import (
    CharSamplerEngineResource,
    CharSamplerEngineRunConfig,
)

logger = logging.getLogger(__name__)


@attrs.define
class CharAndFontSamplerEngineRunConfig:
    height: int
    width: int
    glyph_sequence: FontEngineRunConfigGlyphSequence = \
        FontEngineRunConfigGlyphSequence.HORI_DEFAULT
    num_chars_factor: float = 1.1


@attrs.define
class CharAndFontSamplerEngineResource:
    lexicon_collection: LexiconCollection
    font_collection: FontCollection
    char_sampler_aggregator: EngineRunnerAggregator[
        CharSamplerEngineResource,
        CharSamplerEngineRunConfig,
        Sequence[str],
    ]  # yapf: disable


@attrs.define
class CharAndFont:
    chars: Sequence[str]
    font_variant: FontVariant


class CharAndFontSamplerEngine(
    Engine[
        NoneTypeEngineConfig,
        CharAndFontSamplerEngineResource,
        CharAndFontSamplerEngineRunConfig,
        Optional[CharAndFont],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'default'

    def __init__(
        self,
        config: NoneTypeEngineConfig,
        resource: Optional[CharAndFontSamplerEngineResource] = None,
    ):
        super().__init__(config, resource)

        assert resource
        self.font_collection = resource.font_collection
        self.lexicon_collection = resource.lexicon_collection
        self.char_sampler_aggregator = resource.char_sampler_aggregator

    @staticmethod
    def estimate_num_chars(config: CharAndFontSamplerEngineRunConfig):
        if config.glyph_sequence == FontEngineRunConfigGlyphSequence.HORI_DEFAULT:
            num_chars = config.width / config.height
        elif config.glyph_sequence == FontEngineRunConfigGlyphSequence.VERT_DEFAULT:
            num_chars = config.height / config.width
        else:
            raise NotImplementedError()

        num_chars *= config.num_chars_factor
        return math.ceil(num_chars)

    def run(
        self,
        config: CharAndFontSamplerEngineRunConfig,
        rng: RandomGenerator,
    ) -> Optional[CharAndFont]:
        # Sample chars.
        num_chars = CharAndFontSamplerEngine.estimate_num_chars(config)
        chars = self.char_sampler_aggregator.run(
            CharSamplerEngineRunConfig(num_chars=num_chars),
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
        logger.debug(f'font_variant={font_variant}')

        return CharAndFont(chars=chars, font_variant=font_variant)


char_and_font_sampler_factory = EngineFactory(CharAndFontSamplerEngine)
