from vkit.engine.interface import EngineExecutorAggregatorFactory

from .type import CharSamplerEngineRunConfig
from .func_collate import char_sampler_func_collate

from .corpus import (
    char_sampler_corpus_engine_executor_factory,
    CharSamplerCorpusEngineInitConfig,
    CharSamplerCorpusEngineInitResource,
    CharSamplerCorpusEngine,
)
from .datetime import (
    char_sampler_datetime_engine_executor_factory,
    CharSamplerDatetimeEngineInitConfig,
    CharSamplerDatetimeEngineInitResource,
    CharSamplerDatetimeEngine,
)
from .faker import (
    char_sampler_faker_engine_executor_factory,
    CharSamplerFakerEngineInitConfig,
    CharSamplerFakerEngineInitResource,
    CharSamplerFakerEngine,
)
from .lexicon import (
    char_sampler_lexicon_engine_executor_factory,
    CharSamplerLexiconEngineInitConfig,
    CharSamplerLexiconEngineInitResource,
    CharSamplerLexiconEngine,
)

char_sampler_engine_executor_aggregator_factory = EngineExecutorAggregatorFactory(
    [
        char_sampler_corpus_engine_executor_factory,
        char_sampler_datetime_engine_executor_factory,
        char_sampler_faker_engine_executor_factory,
        char_sampler_lexicon_engine_executor_factory,
    ],
    func_collate=char_sampler_func_collate,
)
