from typing import List

from numpy.random import default_rng
import pytest

from vkit.element import Painter, Image
from vkit.element import LexiconCollection
from vkit.engine.font import font_factory, FontCollection, TextLine
from vkit.engine.char_sampler import char_sampler_factory
from vkit.engine.char_and_font_sampler import char_and_font_sampler_factory
from vkit.engine.seal_impression import *
from tests.opt import write_image


@pytest.mark.local
def test_ellipse_layout():
    engine = ellipse_seal_impression_factory.create({
        'icon_image_folders': [
            '$VKIT_ARTIFACT_PACK/non_text_symbol_image/material-design-icon',
            '$VKIT_ARTIFACT_PACK/non_text_symbol_image/noto-regular-emoji',
        ],
    })

    for rng_seed in range(10):
        # for rng_seed in (8,):
        seal_impression = engine.run({'height': 400, 'width': 400}, default_rng(rng_seed))

        painter = Painter.create(seal_impression.shape)
        painter.paint_mask(seal_impression.background_mask)
        painter.paint_points(
            [
                char_slot.point_up
                for tls in seal_impression.text_line_slots
                for char_slot in tls.char_slots
            ],
            color='green',
            radius=3,
        )
        painter.paint_points(
            [
                char_slot.point_down
                for tls in seal_impression.text_line_slots
                for char_slot in tls.char_slots
            ],
            color='blue',
            radius=3,
        )
        if seal_impression.internal_text_line_box:
            painter.paint_boxes([seal_impression.internal_text_line_box])
        write_image(f'layout_{rng_seed}.jpg', painter.image)


@pytest.mark.local
def test_ellipse_filling():
    lexicon_collection = LexiconCollection.from_file(
        '$VKIT_ARTIFACT_PACK/lexicon_collection/chinese.json'
    )
    font_collection = FontCollection.from_folder('$VKIT_ARTIFACT_PACK/font_collection')
    char_sampler_aggregator = char_sampler_factory.create(
        # '$VKIT_ARTIFACT_PACK/pipeline/text_detection/char_sampler_configs.json',
        [
            {
                "weight": 1,
                "type": "corpus",
                "config": {
                    "txt_files": ["$VKIT_PRIVATE_DATA/char_sampler/debug.txt"]
                }
            },
        ],
        {
            'lexicon_collection': lexicon_collection,
        },
    )
    char_and_font_sampler = char_and_font_sampler_factory.create(
        {},
        {
            'lexicon_collection': lexicon_collection,
            'font_collection': font_collection,
            'char_sampler_aggregator': char_sampler_aggregator,
        },
    )
    font_aggregator = font_factory.create([
        {
            "type": "freetype_default",
            "weight": 1
        },
        {
            "type": "freetype_monochrome",
            "weight": 1
        },
    ])

    engine = ellipse_seal_impression_factory.create({
        'icon_image_folders': [
            '$VKIT_ARTIFACT_PACK/non_text_symbol_image/material-design-icon',
            '$VKIT_ARTIFACT_PACK/non_text_symbol_image/noto-regular-emoji',
        ],
    })

    for rng_seed in range(10):
        rng = default_rng(rng_seed)
        width = int(rng.integers(200, 600 + 1))
        seal_impression = engine.run({'height': 400, 'width': width}, rng)

        text_lines: List[TextLine] = []
        for text_line_slot in seal_impression.text_line_slots:
            char_and_font = char_and_font_sampler.run(
                config={
                    'height': text_line_slot.text_line_height,
                    'width': 2**32 - 1,
                    'num_chars': len(text_line_slot.char_slots),
                },
                rng=rng,
            )
            assert char_and_font

            text_line = font_aggregator.run(
                config={
                    'height': text_line_slot.text_line_height,
                    'width': 2**32 - 1,
                    'chars': char_and_font.chars,
                    'font_variant': char_and_font.font_variant,
                },
                rng=rng,
            )
            assert text_line
            text_lines.append(text_line)

        internal_text_line = None
        if seal_impression.internal_text_line_box:
            char_and_font = char_and_font_sampler.run(
                config={
                    'height': seal_impression.internal_text_line_box.height,
                    'width': seal_impression.internal_text_line_box.width,
                },
                rng=rng,
            )
            assert char_and_font

            internal_text_line = font_aggregator.run(
                config={
                    'height': seal_impression.internal_text_line_box.height,
                    'width': seal_impression.internal_text_line_box.width,
                    'chars': char_and_font.chars,
                    'font_variant': char_and_font.font_variant,
                },
                rng=rng,
            )

        filled_score_map = fill_text_line_to_seal_impression(
            seal_impression,
            list(range(len(text_lines))),
            text_lines,
            internal_text_line,
        )

        image = Image.from_shape(seal_impression.shape)
        seal_impression.background_mask.fill_image(
            image,
            value=seal_impression.color,
            alpha=seal_impression.alpha,
        )
        image[filled_score_map] = seal_impression.color

        write_image(f'{rng_seed}.jpg', image)
