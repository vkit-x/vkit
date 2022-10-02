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
from typing import Sequence, Mapping, Any, List, Union, Optional
from enum import Enum, unique
import logging

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import Box, LexiconCollection
from vkit.utility import (
    normalize_to_keys_and_probs,
    rng_choice,
    PathType,
)
from vkit.engine.font import (
    font_engine_executor_aggregator_factory,
    FontEngineRunConfigStyle,
    FontCollection,
    TextLine,
)
from vkit.engine.char_sampler import char_sampler_engine_executor_aggregator_factory
from vkit.engine.char_and_font_sampler import char_and_font_sampler_engine_executor_factory
from vkit.engine.seal_impression import SealImpression
from ..interface import PipelineStep, PipelineStepFactory
from .page_layout import PageLayoutStepOutput
from .page_seal_impression import PageSealImpresssionStepOutput

logger = logging.getLogger(__name__)


@attrs.define
class PageTextLineStepConfig:
    lexicon_collection_json: str
    font_collection_folder: str
    char_sampler_configs: Union[Sequence[Mapping[str, Any]], PathType]
    font_configs: Union[Sequence[Mapping[str, Any]], PathType]
    font_style: FontEngineRunConfigStyle = attrs.field(factory=FontEngineRunConfigStyle)
    weight_font_style_glyph_color_grayscale: float = 0.9
    font_style_glyph_color_grayscale_min: int = 0
    font_style_glyph_color_grayscale_max: int = 75
    weight_font_style_glyph_color_red: float = 0.04
    weight_font_style_glyph_color_green: float = 0.02
    weight_font_style_glyph_color_blue: float = 0.04
    font_style_glyph_color_rgb_min: int = 128
    font_style_glyph_color_rgb_max: int = 255
    return_font_variant: bool = False
    short_text_line_char_sampler_configs: Optional[
        Union[Sequence[Mapping[str, Any]], PathType]
    ] = None  # yapf: disable
    prob_short_text_line: float = 0.2
    short_text_line_num_chars_max: int = 2


@attrs.define
class PageTextLineStepInput:
    page_layout_step_output: PageLayoutStepOutput
    page_seal_impresssion_step_output: PageSealImpresssionStepOutput


@unique
class PageTextLineStepKey(Enum):
    FONT_STYLE_GLYPH_COLOR_GRAYSCALE = 'font_style_glyph_color_grayscale'
    FONT_STYLE_GLYPH_COLOR_RED = 'font_style_glyph_color_red'
    FONT_STYLE_GLYPH_COLOR_GREEN = 'font_style_glyph_color_green'
    FONT_STYLE_GLYPH_COLOR_BLUE = 'font_style_glyph_color_blue'


@attrs.define
class PageTextLineCollection:
    height: int
    width: int
    text_lines: Sequence[TextLine]
    short_text_line_flags: Sequence[bool]

    @property
    def shape(self):
        return self.height, self.width


@attrs.define
class SealImpressionResource:
    box: Box
    angle: int
    text_line_slot_indices: Sequence[int]
    text_lines: Sequence[TextLine]
    internal_text_line: Optional[TextLine]


@attrs.define
class PageSealImpressionTextLineCollection:
    height: int
    width: int
    seal_impressions: Sequence[SealImpression]
    seal_impression_resources: Sequence[SealImpressionResource]


@attrs.define
class PageTextLineStepOutput:
    page_text_line_collection: PageTextLineCollection
    page_seal_impression_text_line_collection: PageSealImpressionTextLineCollection


class PageTextLineStep(
    PipelineStep[
        PageTextLineStepConfig,
        PageTextLineStepInput,
        PageTextLineStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageTextLineStepConfig):
        super().__init__(config)

        lexicon_collection = LexiconCollection.from_file(self.config.lexicon_collection_json)
        font_collection = FontCollection.from_folder(self.config.font_collection_folder)
        char_sampler_engine_executor_aggregator = \
            char_sampler_engine_executor_aggregator_factory.create_with_repeated_init_resource(
                self.config.char_sampler_configs,
                {
                    'lexicon_collection': lexicon_collection,
                },
            )

        self.char_and_font_sampler_engine_executor = \
            char_and_font_sampler_engine_executor_factory.create(
                {},
                {
                    'lexicon_collection': lexicon_collection,
                    'font_collection': font_collection,
                    'char_sampler_engine_executor_aggregator':
                        char_sampler_engine_executor_aggregator,
                },
            )

        self.short_text_line_char_and_font_sampler_engine_executor = \
            self.char_and_font_sampler_engine_executor

        if self.config.short_text_line_char_sampler_configs is not None:
            short_text_line_char_sampler_engine_executor_aggregator = \
                char_sampler_engine_executor_aggregator_factory.create_with_repeated_init_resource(
                    self.config.short_text_line_char_sampler_configs,
                    {
                        'lexicon_collection': lexicon_collection,
                    },
                )
            self.short_text_line_char_and_font_sampler_engine_executor = \
                char_and_font_sampler_engine_executor_factory.create(
                    {},
                    {
                        'lexicon_collection': lexicon_collection,
                        'font_collection': font_collection,
                        'char_sampler_engine_executor_aggregator':
                            short_text_line_char_sampler_engine_executor_aggregator,
                    },
                )

        self.keys, self.probs = normalize_to_keys_and_probs([
            (
                PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_GRAYSCALE,
                self.config.weight_font_style_glyph_color_grayscale,
            ),
            (
                PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_RED,
                self.config.weight_font_style_glyph_color_red,
            ),
            (
                PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_GREEN,
                self.config.weight_font_style_glyph_color_green,
            ),
            (
                PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_BLUE,
                self.config.weight_font_style_glyph_color_blue,
            ),
        ])
        self.font_engine_executor_aggregator = font_engine_executor_aggregator_factory.create(
            self.config.font_configs
        )

    def run(self, input: PageTextLineStepInput, rng: RandomGenerator):
        page_layout_step_output = input.page_layout_step_output
        page_layout = page_layout_step_output.page_layout

        # Text lines to be recognized.
        text_lines: List[TextLine] = []
        short_text_line_flags: List[bool] = []

        for layout_text_line in page_layout.layout_text_lines:
            char_and_font = None
            is_short_text_line = False

            num_retries = 3
            while num_retries > 0:
                is_short_text_line = (rng.random() < self.config.prob_short_text_line)

                if is_short_text_line:
                    char_and_font_sampler_engine_executor = \
                        self.short_text_line_char_and_font_sampler_engine_executor
                else:
                    char_and_font_sampler_engine_executor = \
                        self.char_and_font_sampler_engine_executor

                char_and_font = char_and_font_sampler_engine_executor.run(
                    run_config={
                        'height': layout_text_line.box.height,
                        'width': layout_text_line.box.width,
                    },
                    rng=rng,
                )
                if char_and_font:
                    break

                num_retries -= 1

            if num_retries <= 0:
                logger.warning(
                    f'Cannot sample char_and_font for layout_text_line={layout_text_line}'
                )
                continue
            assert char_and_font

            if is_short_text_line:
                # Trim to short text line.
                short_text_line_num_chars = int(
                    rng.integers(
                        1,
                        self.config.short_text_line_num_chars_max + 1,
                    )
                )
                chars = [char for char in char_and_font.chars if not char.isspace()]
                if len(chars) > short_text_line_num_chars:
                    begin = int(rng.integers(
                        0,
                        len(chars) - short_text_line_num_chars + 1,
                    ))
                    end = begin + short_text_line_num_chars - 1
                    chars = chars[begin:end + 1]

                logger.debug(f'short_text_line: trim chars={char_and_font.chars} to {chars}.')
                char_and_font = attrs.evolve(char_and_font, chars=chars)

            key = rng_choice(rng, self.keys, probs=self.probs)
            if key == PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_GRAYSCALE:
                grayscale_value = int(
                    rng.integers(
                        self.config.font_style_glyph_color_grayscale_min,
                        self.config.font_style_glyph_color_grayscale_max + 1,
                    )
                )
                glyph_color = (grayscale_value,) * 3

            else:
                rgb_value = int(
                    rng.integers(
                        self.config.font_style_glyph_color_rgb_min,
                        self.config.font_style_glyph_color_rgb_max + 1,
                    )
                )

                if key == PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_RED:
                    glyph_color = (rgb_value, 0, 0)
                elif key == PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_GREEN:
                    glyph_color = (0, rgb_value, 0)
                elif key == PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_BLUE:
                    glyph_color = (0, 0, rgb_value)
                else:
                    raise NotImplementedError()

            font_style = attrs.evolve(
                self.config.font_style,
                glyph_color=glyph_color,
            )
            text_line = self.font_engine_executor_aggregator.run(
                run_config={
                    'height': layout_text_line.box.height,
                    'width': layout_text_line.box.width,
                    'chars': char_and_font.chars,
                    'font_variant': char_and_font.font_variant,
                    'glyph_sequence': layout_text_line.glyph_sequence,
                    'style': font_style,
                    'return_font_variant': self.config.return_font_variant,
                },
                rng=rng,
            )
            if text_line:
                text_line = text_line.to_shifted_text_line(
                    offset_y=layout_text_line.box.up,
                    offset_x=layout_text_line.box.left,
                )
                split_text_lines = text_line.split()
                text_lines.extend(split_text_lines)
                short_text_line_flags.extend([is_short_text_line] * len(split_text_lines))

        assert text_lines
        assert len(text_lines) == len(short_text_line_flags)
        page_text_line_collection = PageTextLineCollection(
            height=page_layout.height,
            width=page_layout.width,
            text_lines=text_lines,
            short_text_line_flags=short_text_line_flags,
        )

        # Text lines for seal impressions.
        page_seal_impresssion_step_output = input.page_seal_impresssion_step_output

        seal_impressions: List[SealImpression] = []
        seal_impression_resources: List[SealImpressionResource] = []

        for seal_impression, box, angle in zip(
            page_seal_impresssion_step_output.seal_impressions,
            page_seal_impresssion_step_output.boxes,
            page_seal_impresssion_step_output.angles,
        ):
            text_line_slot_indices: List[int] = []
            text_lines: List[TextLine] = []

            for text_line_slot_idx, text_line_slot in enumerate(seal_impression.text_line_slots):
                char_and_font = None

                num_retries = 3
                while num_retries > 0:
                    char_and_font = self.char_and_font_sampler_engine_executor.run(
                        run_config={
                            'height': text_line_slot.text_line_height,
                            'width': 2**32 - 1,
                            'num_chars': len(text_line_slot.char_slots),
                        },
                        rng=rng,
                    )
                    if char_and_font:
                        break
                    num_retries -= 1

                if num_retries <= 0:
                    logger.warning(
                        f'Cannot sample char_and_font for seal_impression={seal_impression}'
                    )
                    continue
                assert char_and_font

                text_line = self.font_engine_executor_aggregator.run(
                    run_config={
                        'height': text_line_slot.text_line_height,
                        'width': 2**32 - 1,
                        'chars': char_and_font.chars,
                        'font_variant': char_and_font.font_variant,
                    },
                    rng=rng,
                )
                if text_line:
                    text_line_slot_indices.append(text_line_slot_idx)
                    text_lines.append(text_line)

            internal_text_line = None
            if seal_impression.internal_text_line_box:
                char_and_font = None

                num_retries = 3
                while num_retries > 0:
                    char_and_font = self.char_and_font_sampler_engine_executor.run(
                        run_config={
                            'height': seal_impression.internal_text_line_box.height,
                            'width': seal_impression.internal_text_line_box.width,
                        },
                        rng=rng,
                    )
                    if char_and_font:
                        break
                    num_retries -= 1

                if num_retries <= 0:
                    logger.warning(
                        f'Cannot sample char_and_font for seal_impression={seal_impression}'
                    )
                else:
                    assert char_and_font

                    internal_text_line = self.font_engine_executor_aggregator.run(
                        run_config={
                            'height': seal_impression.internal_text_line_box.height,
                            'width': seal_impression.internal_text_line_box.width,
                            'chars': char_and_font.chars,
                            'font_variant': char_and_font.font_variant,
                        },
                        rng=rng,
                    )

            if text_lines:
                seal_impressions.append(seal_impression)
                seal_impression_resources.append(
                    SealImpressionResource(
                        box=box,
                        angle=angle,
                        text_line_slot_indices=text_line_slot_indices,
                        text_lines=text_lines,
                        internal_text_line=internal_text_line,
                    )
                )

        page_seal_impression_text_line_collection = PageSealImpressionTextLineCollection(
            height=page_layout.height,
            width=page_layout.width,
            seal_impressions=seal_impressions,
            seal_impression_resources=seal_impression_resources,
        )

        return PageTextLineStepOutput(
            page_text_line_collection=page_text_line_collection,
            page_seal_impression_text_line_collection=page_seal_impression_text_line_collection,
        )


page_text_line_step_factory = PipelineStepFactory(PageTextLineStep)
