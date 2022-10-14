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
from typing import Sequence, List, Mapping, Optional
from collections import OrderedDict

import attrs
from numpy.random import Generator as RandomGenerator
from faker import Faker

from vkit.utility import rng_choice, normalize_to_probs
from vkit.engine.interface import Engine, EngineExecutorFactory
from .type import CharSamplerEngineInitResource, CharSamplerEngineRunConfig


@attrs.define
class CharSamplerFakerEngineInitConfig:
    local_to_weight: Mapping[str, float] = {
        'zh_CN': 4,
        'zh_TW': 1,
        'en_US': 5,
    }
    method_to_weight: Mapping[str, float] = {
        'address': 1,
        'ascii_email': 1,
        'dga': 1,
        'uri': 1,
        'word': 10,
        'name': 1,
        'country_calling_code': 1,
        'phone_number': 1,
    }


CharSamplerFakerEngineInitResource = CharSamplerEngineInitResource


class CharSamplerFakerEngine(
    Engine[
        CharSamplerFakerEngineInitConfig,
        CharSamplerFakerEngineInitResource,
        CharSamplerEngineRunConfig,
        Sequence[str],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'faker'

    def __init__(
        self,
        init_config: CharSamplerFakerEngineInitConfig,
        init_resource: Optional[CharSamplerFakerEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        assert init_resource
        self.lexicon_collection = init_resource.lexicon_collection

        self.methods = sorted(init_config.method_to_weight)
        self.methods_probs = normalize_to_probs([
            init_config.method_to_weight[method] for method in self.methods
        ])

        self.faker: Optional[Faker] = None

    def sample_from_faker(self, rng: RandomGenerator):
        # Faker is not picklable, hence need a lazy initialization.
        if self.faker is None:
            self.faker = Faker(OrderedDict(self.init_config.local_to_weight))

        while True:
            method = rng_choice(rng, self.methods, probs=self.methods_probs)
            seed: int = rng.bit_generator.state['state']['state']
            for local in self.init_config.local_to_weight:
                self.faker[local].seed(seed)

            text = getattr(self.faker, method)()
            segments: List[str] = []
            for segment in text.split():
                segment = ''.join(
                    char for char in segment if self.lexicon_collection.has_char(char)
                )
                if segment:
                    segments.append(segment)
            if segments:
                return ' '.join(segments)

    def run(self, run_config: CharSamplerEngineRunConfig, rng: RandomGenerator) -> Sequence[str]:
        if not run_config.enable_aggregator_mode:
            num_chars = run_config.num_chars

            texts: List[str] = []
            num_chars_in_texts = 0
            while num_chars_in_texts + len(texts) - 1 < num_chars:
                text = self.sample_from_faker(rng)
                texts.append(text)
                num_chars_in_texts += len(text)

            chars = list(' '.join(texts))

            # Trim and make sure the last char is not space.
            if len(chars) > num_chars:
                rest = chars[num_chars:]
                chars = chars[:num_chars]
                if chars[-1].isspace():
                    chars.pop()
                    assert not rest[0].isspace()
                    chars.append(rest[0])

            return chars

        else:
            return self.sample_from_faker(rng)


char_sampler_faker_engine_executor_factory = EngineExecutorFactory(CharSamplerFakerEngine)
