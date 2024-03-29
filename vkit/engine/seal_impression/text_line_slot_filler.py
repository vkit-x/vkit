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
from typing import Sequence, List, Optional
import logging

import numpy as np
import attrs

from vkit.element import Point, Box, Polygon, ScoreMap
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
    char_polygons: List[Polygon] = []

    assert len(text_line_slot_indices) == len(text_lines)

    for text_line_slot_idx, text_line in zip(text_line_slot_indices, text_lines):
        if text_line_slot_idx >= len(seal_impression.text_line_slots):
            logger.error('fill_text_line_to_seal_impression: something wrong.')
            break

        assert text_line.is_hori
        assert not text_line.shifted

        # Get the text line slot to be filled.
        text_line_slot = seal_impression.text_line_slots[text_line_slot_idx]

        # Get the reference char height of text line, for adjusting aspect ratio.
        text_line_ref_char_height = 0
        text_line_ref_char_width = 0
        for char_glyph in text_line.char_glyphs:
            if char_glyph.ref_char_height > text_line_ref_char_height:
                text_line_ref_char_height = char_glyph.ref_char_height
                text_line_ref_char_width = char_glyph.ref_char_width
        assert text_line_ref_char_height > 0 and text_line_ref_char_width > 0
        text_line_aspect_ratio = text_line_ref_char_width / text_line_ref_char_height

        # For resizing.
        resized_char_width_factor = text_line_slot.char_aspect_ratio / text_line_aspect_ratio

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
            resized_width = max(1, round(resized_char_width_factor * char_glyph.width))
            resized_box = attrs.evolve(char_box.box, left=0, right=resized_width - 1)
            # NOTE: Since the height of char is fixed, we simply preserve the text line height.
            char_score_map = ScoreMap.from_shape((text_line.box.height, resized_width))

            if char_glyph.score_map:
                char_glyph_score_map = char_glyph.score_map
                if char_glyph_score_map.shape != resized_box.shape:
                    char_glyph_score_map = char_glyph_score_map.to_resized_score_map(
                        resized_height=resized_box.height,
                        resized_width=resized_box.width,
                        cv_resize_interpolation=text_line.cv_resize_interpolation,
                    )
                resized_box.fill_score_map(char_score_map, char_glyph_score_map)

            else:
                # LCD, fallback to mask.
                char_glyph_mask = char_glyph.get_glyph_mask(
                    box=char_box.box,
                    cv_resize_interpolation=text_line.cv_resize_interpolation,
                )
                if char_glyph_mask.shape != resized_box.shape:
                    char_glyph_mask = char_glyph_mask.to_resized_mask(
                        resized_height=resized_box.height,
                        resized_width=resized_box.width,
                        cv_resize_interpolation=text_line.cv_resize_interpolation,
                    )
                resized_box.fill_score_map(char_score_map, char_glyph_mask.mat.astype(np.float32))

            # To match char_slot.point_up.
            point_up = Point.create(y=0, x=char_score_map.width / 2)

            # Generate char polygon.
            up = resized_box.up
            down = resized_box.down
            ref_char_height = char_glyph.ref_char_height
            if resized_box.height < ref_char_height:
                inc = ref_char_height - resized_box.height
                half_inc = inc / 2
                up = up - half_inc
                down = down + half_inc

            left = resized_box.left
            right = resized_box.right
            # NOTE: Resize factor is applied.
            ref_char_width = resized_char_width_factor * char_glyph.ref_char_width
            if resized_box.width < ref_char_width:
                inc = ref_char_width - resized_box.width
                half_inc = inc / 2
                left = left - half_inc
                right = right + half_inc

            char_polygon = Polygon.from_xy_pairs([
                (left, up),
                (right, up),
                (right, down),
                (left, down),
            ])

            # Rotate.
            rotated_result = rotate.distort(
                # Horizontal text line has angle 270.
                {'angle': char_slot.angle - 270},
                score_map=char_score_map,
                point=point_up,
                polygon=char_polygon,
                # The char polygon could be out-of-bound and should not be clipped.
                disable_clip_result_elements=True,
            )
            rotated_char_score_map = rotated_result.score_map
            assert rotated_char_score_map
            rotated_point_up = rotated_result.point
            assert rotated_point_up
            rotated_char_polygon = rotated_result.polygon
            assert rotated_char_polygon

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

            # Shift and keep rotated char polygon.
            rotated_char_polygon = rotated_char_polygon.to_shifted_polygon(
                offset_y=dst_up,
                offset_x=dst_left,
            )
            char_polygons.append(rotated_char_polygon)

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

        char_polygons.extend(
            internal_text_line.to_char_polygons(
                page_height=score_map.height,
                page_width=score_map.width,
            )
        )

    # Adjust alpha.
    score_map_max = score_map.mat.max()
    score_map.assign_mat(score_map.mat * seal_impression.alpha / score_map_max)

    return score_map, char_polygons
