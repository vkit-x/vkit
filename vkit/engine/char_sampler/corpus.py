from typing import Sequence, List, Optional

import attrs
import iolite as io
from numpy.random import Generator as RandomGenerator

from vkit.utility import rng_choice
from vkit.engine.interface import Engine
from .type import CharSamplerEngineResource, CharSamplerEngineRunConfig


@attrs.define
class CorpusCharSamplerEngineConfig:
    txt_files: Sequence[str]


class CorpusCharSamplerEngine(
    Engine[
        CorpusCharSamplerEngineConfig,
        CharSamplerEngineResource,
        CharSamplerEngineRunConfig,
        Sequence[str],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'corpus'

    def __init__(
        self,
        config: CorpusCharSamplerEngineConfig,
        resource: Optional[CharSamplerEngineResource] = None
    ):
        super().__init__(config, resource)

        assert resource
        self.lexicon_collection = resource.lexicon_collection

        self.texts: List[str] = []
        for txt_file in config.txt_files:
            for line in io.read_text_lines(
                txt_file,
                expandvars=True,
                strip=True,
                skip_empty=True,
            ):
                self.texts.append(line)

    def sample_and_prep_text(self, rng: RandomGenerator):
        while True:
            text = rng_choice(rng, self.texts)
            segments: List[str] = []
            for segment in text.split():
                segment = ''.join(
                    char for char in segment if self.lexicon_collection.has_char(char)
                )
                if segment:
                    segments.append(segment)
            if segments:
                return ' '.join(segments)

    def run(self, config: CharSamplerEngineRunConfig, rng: RandomGenerator) -> Sequence[str]:
        num_chars = config.num_chars
        if num_chars <= 0:
            return []

        # Uniform selection.
        texts: List[str] = []
        num_chars_in_texts = 0
        while num_chars_in_texts + len(texts) - 1 < num_chars:
            text = self.sample_and_prep_text(rng)
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
