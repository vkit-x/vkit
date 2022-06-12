from vkit.engine.interface import EngineFactory, EngineRunnerAggregatorFactory
from .type import CharSamplerEngineResource, CharSamplerEngineRunConfig
from .corpus import CorpusCharSamplerEngine, CorpusCharSamplerEngineConfig
from .datetime import DatetimeCharSamplerEngine, DatetimeCharSamplerEngineConfig
from .faker import FakerCharSamplerEngine, FakerCharSamplerEngineConfig
from .lexicon import LexiconCharSamplerEngine, LexiconCharSamplerEngineConfig

corpus_char_sampler_factory = EngineFactory(CorpusCharSamplerEngine)
datetime_char_sampler_factory = EngineFactory(DatetimeCharSamplerEngine)
faker_char_sampler_factory = EngineFactory(FakerCharSamplerEngine)
lexicon_char_sampler_factory = EngineFactory(LexiconCharSamplerEngine)

char_sampler_factory = EngineRunnerAggregatorFactory([
    corpus_char_sampler_factory,
    datetime_char_sampler_factory,
    faker_char_sampler_factory,
    lexicon_char_sampler_factory,
])
