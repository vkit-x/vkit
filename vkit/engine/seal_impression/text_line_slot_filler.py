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
            logger.error('something wrong.')
            break

        assert text_line.is_hori
        assert not text_line.shifted

        ref_char_height = 0
        ref_char_width = 0
        for char_glyph in text_line.char_glyphs:
            if char_glyph.ref_char_height > ref_char_height:
                ref_char_height = char_glyph.ref_char_height
                ref_char_width = char_glyph.ref_char_width
        assert ref_char_height > 0 and ref_char_width > 0

        text_line_slot = seal_impression.text_line_slots[text_line_slot_idx]
        char_aspect_ratio = text_line_slot.char_aspect_ratio
        text_line_aspect_ratio = ref_char_width / ref_char_height
        resized_width_factor = char_aspect_ratio / text_line_aspect_ratio

        for char_slot_idx, (char_box, char_glyph) \
                in enumerate(zip(text_line.char_boxes, text_line.char_glyphs)):
            # Get char slot to be filled.
            if char_slot_idx >= len(text_line_slot.char_slots):
                logger.warning('something wrong.')
                break
            char_slot = text_line_slot.char_slots[char_slot_idx]

            # Convert char glyph to score map.
            box = char_box.box
            resized_width = max(1, round(resized_width_factor * box.width))
            char_glyph_shape = (box.height, resized_width)
            char_score_map = ScoreMap.from_shape((text_line.box.height, resized_width))

            if char_glyph.score_map:
                char_glyph_score_map = char_glyph.score_map
                if char_glyph_score_map.shape != char_glyph_shape:
                    char_glyph_score_map = char_glyph_score_map.to_resized_score_map(
                        resized_height=char_glyph_shape[0],
                        resized_width=char_glyph_shape[1],
                        cv_resize_interpolation=text_line.cv_resize_interpolation,
                    )
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
                with char_score_map.writable_context:
                    char_score_map.mat[box.up:box.down + 1] = char_glyph_mask.mat.astype(np.float32)

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

            # Shift bounding box based on point_up.
            dst_up = char_slot.point_up.y - rotated_point_up.y
            dst_down = char_slot.point_up.y + (
                rotated_char_score_map.height - 1 - rotated_point_up.y
            )
            dst_left = char_slot.point_up.x - rotated_point_up.x
            dst_right = char_slot.point_up.x + (
                rotated_char_score_map.width - 1 - rotated_point_up.x
            )

            # Corner case: out-of-bound.
            src_up = 0
            if dst_up < 0:
                src_up = abs(dst_up)
                dst_up = 0

            src_down = rotated_char_score_map.height - 1
            if dst_down >= score_map.height:
                src_down -= dst_down + 1 - score_map.height
                dst_down = score_map.height - 1

            assert src_up <= src_down and dst_up <= dst_down
            assert src_down - src_up == dst_down - dst_up

            src_left = 0
            if dst_left < 0:
                src_left = abs(dst_left)
                dst_left = 0

            src_right = rotated_char_score_map.width - 1
            if dst_right >= score_map.width:
                src_right -= dst_right + 1 - score_map.width
                dst_right = score_map.width - 1

            assert src_left <= src_right and dst_left <= dst_right
            assert src_right - src_left == dst_right - dst_left

            # Fill.
            src_box = Box(up=src_up, down=src_down, left=src_left, right=src_right)
            dst_box = Box(up=dst_up, down=dst_down, left=dst_left, right=dst_right)
            dst_box.fill_score_map(
                score_map,
                src_box.extract_score_map(rotated_char_score_map),
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
