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
from typing import Sequence, Mapping, Optional, List

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import rng_choice, normalize_to_probs
from vkit.engine.interface import Engine, EngineExecutorFactory
from .type import CharSamplerEngineInitResource, CharSamplerEngineRunConfig


@attrs.define
class CharSamplerLexiconEngineInitConfig:
    tag_to_weight: Optional[Mapping[str, float]] = None
    prob_space: float = 0.0


CharSamplerLexiconEngineInitResource = CharSamplerEngineInitResource


class CharSamplerLexiconEngine(
    Engine[
        CharSamplerLexiconEngineInitConfig,
        CharSamplerLexiconEngineInitResource,
        CharSamplerEngineRunConfig,
        Sequence[str],
    ]
):  # yapf: disable

    KEY_SPACE = '__space'

    @classmethod
    def get_type_name(cls) -> str:
        return 'lexicon'

    def __init__(
        self,
        init_config: CharSamplerLexiconEngineInitConfig,
        init_resource: Optional[CharSamplerLexiconEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        assert init_resource
        self.lexicon_collection = init_resource.lexicon_collection

        tag_weights = []
        for tag in self.lexicon_collection.tags:
            if init_config.tag_to_weight:
                # From config.
                if tag not in init_config.tag_to_weight:
                    raise RuntimeError(f'missing tag={tag} in tag_to_weight')
                weight = init_config.tag_to_weight[tag]
            else:
                # Based on the number of tagged lexicons.
                weight = len(self.lexicon_collection.tag_to_lexicons[tag])
            tag_weights.append(weight)

        self.tags = self.lexicon_collection.tags
        self.tag_probs = normalize_to_probs(tag_weights)

        self.with_space_tags = self.tags
        self.with_space_tag_probs = self.tag_probs
        if init_config.prob_space > 0.0:
            self.with_space_tags = (*self.tags, self.KEY_SPACE)
            self.with_space_tag_probs = normalize_to_probs((
                *self.tag_probs,
                init_config.prob_space / (1 - init_config.prob_space),
            ))

    def run(self, run_config: CharSamplerEngineRunConfig, rng: RandomGenerator) -> Sequence[str]:
        num_chars = run_config.num_chars

        if run_config.enable_aggregator_mode:
            num_chars = int(rng.integers(1, run_config.num_chars + 1))

        chars: List[str] = []
        for char_idx in range(num_chars):
            tag = rng_choice(rng, self.with_space_tags, probs=self.with_space_tag_probs)
            if tag == self.KEY_SPACE:
                if char_idx == 0 \
                        or char_idx == num_chars - 1 \
                        or chars[char_idx - 1].isspace():
                    # Disallow:
                    # 1. leading or trailing space.
                    # 2. consecutive spaces.
                    tag = rng_choice(rng, self.tags, probs=self.tag_probs)

            if tag == self.KEY_SPACE:
                chars.append(' ')
            else:
                lexicon = rng_choice(rng, self.lexicon_collection.tag_to_lexicons[tag])
                char = rng_choice(rng, lexicon.char_and_aliases)
                chars.append(char)

        return chars


char_sampler_lexicon_engine_executor_factory = EngineExecutorFactory(CharSamplerLexiconEngine)
