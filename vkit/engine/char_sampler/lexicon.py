from typing import Sequence, Mapping, Optional, List

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import rng_choice
from vkit.engine.interface import Engine
from .type import CharSamplerEngineResource, CharSamplerEngineRunConfig


@attrs.define
class LexiconCharSamplerEngineConfig:
    tag_to_weight: Optional[Mapping[str, float]] = None
    space_prob: float = 0.0


class LexiconCharSamplerEngine(
    Engine[
        LexiconCharSamplerEngineConfig,
        CharSamplerEngineResource,
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
        config: LexiconCharSamplerEngineConfig,
        resource: Optional[CharSamplerEngineResource] = None,
    ):
        super().__init__(config, resource)

        assert resource
        self.lexicon_collection = resource.lexicon_collection

        weights = []
        if config.tag_to_weight:
            for tag in self.lexicon_collection.tags:
                assert tag in config.tag_to_weight
                weights.append(config.tag_to_weight[tag])
        else:
            for tag in self.lexicon_collection.tags:
                weights.append(len(self.lexicon_collection.tag_to_lexicons[tag]))

        self.tags_no_space = list(self.lexicon_collection.tags)
        total = sum(weights)
        self.probs_no_space = [val / total for val in weights]

        self.tags = list(self.tags_no_space)
        if config.space_prob > 0:
            space_weight = total * config.space_prob / (1 - config.space_prob)
            self.tags.append(self.KEY_SPACE)
            weights.append(space_weight)
            total += space_weight
        self.probs = [val / total for val in weights]

    def run(self, config: CharSamplerEngineRunConfig, rng: RandomGenerator) -> Sequence[str]:
        num_chars = config.num_chars

        chars: List[str] = []
        for char_idx in range(num_chars):
            tag = rng_choice(rng, self.tags, probs=self.probs)
            if tag == self.KEY_SPACE and (char_idx == 0 or char_idx == num_chars - 1):
                tag = rng_choice(rng, self.tags_no_space, probs=self.probs_no_space)

            if tag == self.KEY_SPACE:
                chars.append(' ')
            else:
                lexicon = rng_choice(rng, self.lexicon_collection.tag_to_lexicons[tag])
                char = rng_choice(rng, lexicon.char_and_aliases)
                chars.append(char)

        return chars
