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
from typing import Sequence, Tuple, List, Optional
from datetime import date, datetime
import time

import attrs
from numpy.random import Generator as RandomGenerator
import pytz

from vkit.utility import rng_choice
from vkit.engine.interface import Engine, EngineExecutorFactory
from .type import CharSamplerEngineInitResource, CharSamplerEngineRunConfig


@attrs.define
class CharSamplerDatetimeEngineInitConfig:
    datetime_formats: Sequence[str]
    timezones: Sequence[str]
    datetime_begin: Tuple[int, int, int] = (1991, 12, 25)
    datetime_end: Tuple[int, int, int] = (2050, 12, 31)


CharSamplerDatetimeEngineInitResource = CharSamplerEngineInitResource


class CharSamplerDatetimeEngine(
    Engine[
        CharSamplerDatetimeEngineInitConfig,
        CharSamplerDatetimeEngineInitResource,
        CharSamplerEngineRunConfig,
        Sequence[str],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'datetime'

    def __init__(
        self,
        init_config: CharSamplerDatetimeEngineInitConfig,
        init_resource: Optional[CharSamplerDatetimeEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        assert init_resource
        self.lexicon_collection = init_resource.lexicon_collection
        self.delimiters = [
            char for char in ['/', ':', '-', ',', '.', '*']
            if self.lexicon_collection.has_char(char)
        ]
        self.ticks_begin = int(time.mktime(date(*init_config.datetime_begin).timetuple()))
        self.ticks_end = int(time.mktime(date(*init_config.datetime_end).timetuple()))

    def sample_datetime_text(self, rng: RandomGenerator):
        # Datetime.
        ticks = rng.integers(self.ticks_begin, self.ticks_end + 1)
        # I don't know why, but it works.
        dt = datetime.fromtimestamp(ticks)
        tz = pytz.timezone(rng_choice(rng, self.init_config.timezones))
        dt = tz.localize(dt)

        # Datetime format.
        datetime_format = rng_choice(rng, self.init_config.datetime_formats)
        delimiters = [delimiter for delimiter in self.delimiters if delimiter in datetime_format]
        if delimiters:
            selected_delimiter = rng_choice(rng, delimiters)
            other_delimiters = [
                delimiter for delimiter in self.delimiters if delimiter != selected_delimiter
            ]
            other_delimiters.append(' ')
            repl_delimiter = rng_choice(rng, other_delimiters)
            datetime_format = datetime_format.replace(selected_delimiter, repl_delimiter)

        # To text.
        text = dt.strftime(datetime_format)
        return ''.join(
            char for char in text if char.isspace() or self.lexicon_collection.has_char(char)
        ).strip()

    def run(self, run_config: CharSamplerEngineRunConfig, rng: RandomGenerator) -> Sequence[str]:
        if not run_config.enable_aggregator_mode:
            num_chars = run_config.num_chars

            texts: List[str] = []
            num_chars_in_texts = 0
            while num_chars_in_texts + len(texts) - 1 < num_chars:
                text = self.sample_datetime_text(rng)
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
            return self.sample_datetime_text(rng)


char_sampler_datetime_engine_executor_factory = EngineExecutorFactory(CharSamplerDatetimeEngine)
