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
