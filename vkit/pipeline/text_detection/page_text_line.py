from typing import Sequence, Dict, Any, List, Union
from enum import Enum, unique

import attrs
from numpy.random import RandomState

from vkit.element import LexiconCollection
from vkit.utility import (
    normalize_to_keys_and_probs,
    rnd_choice,
    PathType,
)
from vkit.engine.font import (
    font_factory,
    FontEngineRunConfigStyle,
    FontCollection,
    TextLine,
)
from vkit.engine.char_sampler import char_sampler_factory
from vkit.engine.char_and_font_sampler import char_and_font_sampler_factory
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep


@attrs.define
class PageTextLineStepConfig:
    lexicon_collection_json: str
    font_collection_folder: str
    char_sampler_configs: Union[Sequence[Dict[str, Any]], PathType]
    font_configs: Union[Sequence[Dict[str, Any]], PathType]
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

    @property
    def shape(self):
        return self.height, self.width


@attrs.define
class PageTextLineStepOutput:
    page_text_line_collection: PageTextLineCollection


class PageTextLineStep(
    PipelineStep[
        PageTextLineStepConfig,
        PageTextLineStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageTextLineStepConfig):
        super().__init__(config)

        lexicon_collection = LexiconCollection.from_file(self.config.lexicon_collection_json)
        font_collection = FontCollection.from_folder(self.config.font_collection_folder)
        char_sampler_aggregator = char_sampler_factory.create(
            self.config.char_sampler_configs,
            {
                'lexicon_collection': lexicon_collection,
            },
        )
        self.char_and_font_sampler = char_and_font_sampler_factory.create(
            {},
            {
                'lexicon_collection': lexicon_collection,
                'font_collection': font_collection,
                'char_sampler_aggregator': char_sampler_aggregator,
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
        self.font_aggregator = font_factory.create(self.config.font_configs)

    def run(self, state: PipelineState, rnd: RandomState):
        page_layout_step_output = self.get_output(state, PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        text_lines: List[TextLine] = []
        for layout_text_line in page_layout.layout_text_lines:

            while True:
                char_and_font = self.char_and_font_sampler.run(
                    config={
                        'height': layout_text_line.box.height,
                        'width': layout_text_line.box.width,
                    },
                    rnd=rnd,
                )
                if char_and_font:
                    break

            key = rnd_choice(rnd, self.keys, probs=self.probs)
            if key == PageTextLineStepKey.FONT_STYLE_GLYPH_COLOR_GRAYSCALE:
                grayscale_value = rnd.randint(
                    self.config.font_style_glyph_color_grayscale_min,
                    self.config.font_style_glyph_color_grayscale_max + 1,
                )
                glyph_color = (grayscale_value,) * 3

            else:
                rgb_value = rnd.randint(
                    self.config.font_style_glyph_color_rgb_min,
                    self.config.font_style_glyph_color_rgb_max + 1,
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
            text_line = self.font_aggregator.run(
                config={
                    'height': layout_text_line.box.height,
                    'width': layout_text_line.box.width,
                    'chars': char_and_font.chars,
                    'font_variant': char_and_font.font_variant,
                    'glyph_sequence': layout_text_line.glyph_sequence,
                    'style': font_style,
                    'return_font_variant': self.config.return_font_variant,
                },
                rnd=rnd,
            )
            if text_line:
                text_line = text_line.to_shifted_text_line(
                    y_offset=layout_text_line.box.up,
                    x_offset=layout_text_line.box.left,
                )
                text_lines.extend(text_line.split())

        assert text_lines
        return PageTextLineStepOutput(
            page_text_line_collection=PageTextLineCollection(
                height=page_layout.height,
                width=page_layout.width,
                text_lines=text_lines,
            )
        )


page_text_line_step_factory = PipelineStepFactory(PageTextLineStep)
