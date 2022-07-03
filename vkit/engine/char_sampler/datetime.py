from typing import Sequence, Tuple, List, Optional
from datetime import date, datetime
import time

import attrs
from numpy.random import Generator as RandomGenerator
import pytz

from vkit.utility import rng_choice
from vkit.engine.interface import Engine
from .type import CharSamplerEngineResource, CharSamplerEngineRunConfig


@attrs.define
class DatetimeCharSamplerEngineConfig:
    datetime_formats: Sequence[str]
    timezones: Sequence[str]
    datetime_begin: Tuple[int, int, int] = (1991, 12, 25)
    datetime_end: Tuple[int, int, int] = (2050, 12, 31)


class DatetimeCharSamplerEngine(
    Engine[
        DatetimeCharSamplerEngineConfig,
        CharSamplerEngineResource,
        CharSamplerEngineRunConfig,
        Sequence[str],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'datetime'

    def __init__(
        self,
        config: DatetimeCharSamplerEngineConfig,
        resource: Optional[CharSamplerEngineResource] = None,
    ):
        super().__init__(config, resource)

        assert resource
        self.lexicon_collection = resource.lexicon_collection
        self.delimiters = [
            char for char in ['/', ':', '-', ',', '.', '*']
            if self.lexicon_collection.has_char(char)
        ]
        self.ticks_begin = int(time.mktime(date(*config.datetime_begin).timetuple()))
        self.ticks_end = int(time.mktime(date(*config.datetime_end).timetuple()))

    def sample_datetime_text(self, rng: RandomGenerator):
        # Datetime.
        ticks = rng.integers(self.ticks_begin, self.ticks_end + 1)
        # I don't know why, but it works.
        dt = datetime.fromtimestamp(ticks)
        tz = pytz.timezone(rng_choice(rng, self.config.timezones))
        dt = tz.localize(dt)

        # Datetime format.
        datetime_format = rng_choice(rng, self.config.datetime_formats)
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

    def run(self, config: CharSamplerEngineRunConfig, rng: RandomGenerator) -> Sequence[str]:
        num_chars = config.num_chars

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
