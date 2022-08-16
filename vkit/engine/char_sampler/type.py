import attrs

from vkit.element import LexiconCollection


@attrs.define
class CharSamplerEngineInitResource:
    lexicon_collection: LexiconCollection


@attrs.define
class CharSamplerEngineRunConfig:
    num_chars: int
    enable_aggregator_mode: bool = False
