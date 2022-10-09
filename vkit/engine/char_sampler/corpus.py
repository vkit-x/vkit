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
from typing import Sequence, List, Optional, Tuple
from os.path import getsize
import logging
from pathlib import Path

import attrs
from numpy.random import Generator as RandomGenerator
import iolite as io

from vkit.utility import normalize_to_probs, rng_choice
from vkit.engine.interface import Engine, EngineExecutorFactory
from .type import CharSamplerEngineInitResource, CharSamplerEngineRunConfig

logger = logging.getLogger(__name__)


@attrs.define
class CharSamplerCorpusEngineInitConfig:
    txt_files: Sequence[str]


CharSamplerCorpusEngineInitResource = CharSamplerEngineInitResource


class CharSamplerCorpusEngine(
    Engine[
        CharSamplerCorpusEngineInitConfig,
        CharSamplerCorpusEngineInitResource,
        CharSamplerEngineRunConfig,
        Sequence[str],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'corpus'

    def __init__(
        self,
        init_config: CharSamplerCorpusEngineInitConfig,
        init_resource: Optional[CharSamplerCorpusEngineInitResource] = None
    ):
        super().__init__(init_config, init_resource)

        assert init_resource
        self.lexicon_collection = init_resource.lexicon_collection

        self.txt_file_size_pairs: List[Tuple[Path, int]] = []
        for txt_file in init_config.txt_files:
            txt_file = io.file(txt_file, expandvars=True, exists=True)
            self.txt_file_size_pairs.append((
                txt_file,
                getsize(txt_file),
            ))
        self.txt_file_probs = normalize_to_probs([size for _, size in self.txt_file_size_pairs])

    @classmethod
    def sample_text_line_from_file(
        cls,
        txt_file: Path,
        size: int,
        rng: RandomGenerator,
    ):
        pos = int(rng.integers(0, size))
        with txt_file.open('rb') as fin:
            # Find the next newline.
            end = pos + 1
            while end < size:
                fin.seek(end)
                if fin.read(1) == b'\n':
                    break
                end += 1
            # Find the prev newline.
            begin = pos
            while begin >= 0:
                fin.seek(begin)
                if fin.read(1) == b'\n':
                    break
                begin -= 1
            # Read line.
            begin += 1
            fin.seek(begin)
            binary = fin.read(end - begin)
            # Decode.
            try:
                return binary.decode()
            except UnicodeError:
                logger.exception(f'Failed to decode {binary}')
                return ''

    def sample_text_line(self, rng: RandomGenerator):
        txt_file, size = rng_choice(rng, self.txt_file_size_pairs, probs=self.txt_file_probs)
        return self.sample_text_line_from_file(txt_file, size, rng)

    def sample_and_prep_text(self, rng: RandomGenerator):
        while True:
            text = self.sample_text_line(rng)
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

        else:
            return self.sample_and_prep_text(rng)


char_sampler_corpus_engine_executor_factory = EngineExecutorFactory(CharSamplerCorpusEngine)
