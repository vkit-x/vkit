import attrs

from vkit.element import LexiconCollection


@attrs.define
class CharSamplerEngineResource:
    lexicon_collection: LexiconCollection


@attrs.define
class CharSamplerEngineRunConfig:
    num_chars: int
