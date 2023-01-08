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
from typing import Sequence, Optional
import logging

import numpy as np

from vkit.element import Point, Box, ScoreMap
from vkit.engine.font import TextLine
from vkit.mechanism.distortion import rotate
from .type import SealImpression

logger = logging.getLogger(__name__)


def fill_text_line_to_seal_impression(
    seal_impression: SealImpression,
    text_line_slot_indices: Sequence[int],
    text_lines: Sequence[TextLine],
    internal_text_line: Optional[TextLine],
):
    score_map = ScoreMap.from_shape(seal_impression.shape)

    assert len(text_line_slot_indices) == len(text_lines)

    for text_line_slot_idx, text_line in zip(text_line_slot_indices, text_lines):
        if text_line_slot_idx >= len(seal_impression.text_line_slots):
            logger.error('fill_text_line_to_seal_impression: something wrong.')
            break

        assert text_line.is_hori
        assert not text_line.shifted

        # Get the reference char height of text line, for adjusting aspect ratio.
        ref_char_height = 0
        ref_char_width = 0
        for char_glyph in text_line.char_glyphs:
            if char_glyph.ref_char_height > ref_char_height:
                ref_char_height = char_glyph.ref_char_height
                ref_char_width = char_glyph.ref_char_width
        assert ref_char_height > 0 and ref_char_width > 0

        # Get the text line slot to be filled.
        text_line_slot = seal_impression.text_line_slots[text_line_slot_idx]
        char_aspect_ratio = text_line_slot.char_aspect_ratio
        text_line_aspect_ratio = ref_char_width / ref_char_height
        resized_width_factor = char_aspect_ratio / text_line_aspect_ratio

        # Fill each char to the char slot.
        for char_slot_idx, (char_box, char_glyph) \
                in enumerate(zip(text_line.char_boxes, text_line.char_glyphs)):
            # Get the char slot to be filled.
            if char_slot_idx >= len(text_line_slot.char_slots):
                logger.error('fill_text_line_to_seal_impression: something wrong.')
                break

            char_slot = text_line_slot.char_slots[char_slot_idx]

            # Convert char glyph to score map and adjust aspect ratio.
            # NOTE: Only the width of char could be resized.
            box = char_box.box
            resized_width = max(1, round(resized_width_factor * box.width))
            char_glyph_shape = (box.height, resized_width)
            # NOTE: Since the height of char is fixed, we simply preserve the text line height.
            char_score_map = ScoreMap.from_shape((text_line.box.height, resized_width))

            if char_glyph.score_map:
                char_glyph_score_map = char_glyph.score_map
                if char_glyph_score_map.shape != char_glyph_shape:
                    char_glyph_score_map = char_glyph_score_map.to_resized_score_map(
                        resized_height=char_glyph_shape[0],
                        resized_width=char_glyph_shape[1],
                        cv_resize_interpolation=text_line.cv_resize_interpolation,
                    )
                assert char_score_map.width == char_glyph_score_map.width
                with char_score_map.writable_context:
                    char_score_map.mat[box.up:box.down + 1] = char_glyph_score_map.mat

            else:
                # LCD, fallback to mask.
                char_glyph_mask = char_glyph.get_glyph_mask(
                    box=box,
                    cv_resize_interpolation=text_line.cv_resize_interpolation,
                )
                if char_glyph_mask.shape != char_glyph_shape:
                    char_glyph_mask = char_glyph_mask.to_resized_mask(
                        resized_height=char_glyph_shape[0],
                        resized_width=char_glyph_shape[1],
                        cv_resize_interpolation=text_line.cv_resize_interpolation,
                    )
                assert char_score_map.width == char_glyph_mask.width
                with char_score_map.writable_context:
                    char_score_map.mat[box.up:box.down + 1] = char_glyph_mask.mat.astype(np.float32)

            # To match char_slot.point_up.
            point_up = Point.create(y=0, x=char_score_map.width / 2)

            # Rotate.
            rotated_result = rotate.distort(
                # Horizontal text line has angle 270.
                {'angle': char_slot.angle - 270},
                score_map=char_score_map,
                point=point_up,
            )
            rotated_char_score_map = rotated_result.score_map
            assert rotated_char_score_map
            rotated_point_up = rotated_result.point
            assert rotated_point_up

            # Calculate the bounding box based on point_up.
            # NOTE: rotated_point_up.y represents the vertical offset here.
            dst_up = char_slot.point_up.y - rotated_point_up.y
            dst_down = dst_up + rotated_char_score_map.height - 1
            # NOTE: rotated_point_up.x represents the horizontal offset here.
            dst_left = char_slot.point_up.x - rotated_point_up.x
            dst_right = dst_left + rotated_char_score_map.width - 1

            if dst_up < 0 \
                    or dst_down >= score_map.height \
                    or dst_left < 0 \
                    or dst_right >= score_map.width:
                logger.error('fill_text_line_to_seal_impression: out-of-bound.')
                continue

            # Fill.
            dst_box = Box(up=dst_up, down=dst_down, left=dst_left, right=dst_right)
            dst_box.fill_score_map(
                score_map,
                rotated_char_score_map,
                keep_max_value=True,
            )

    if internal_text_line:
        internal_text_line_box = seal_impression.internal_text_line_box
        assert internal_text_line_box

        internal_text_line = internal_text_line.to_shifted_text_line(
            offset_y=internal_text_line_box.up,
            offset_x=internal_text_line_box.left,
        )
        if internal_text_line.score_map:
            internal_text_line.box.fill_score_map(score_map, internal_text_line.score_map)
        else:
            internal_text_line.box.fill_score_map(score_map, internal_text_line.mask.mat)

    # Adjust alpha.
    score_map_max = score_map.mat.max()
    score_map.assign_mat(score_map.mat * seal_impression.alpha / score_map_max)

    return score_map
