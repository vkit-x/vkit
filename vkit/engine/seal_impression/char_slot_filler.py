import numpy as np

from vkit.element import Point, Box, ScoreMap
from vkit.engine.font import TextLine
from vkit.engine.distortion.geometric.affine import rotate
from .type import SealImpressionLayout


def fill_text_line_to_seal_impression_layout(
    seal_impression_layout: SealImpressionLayout,
    text_line: TextLine,
):
    assert text_line.is_hori
    assert not text_line.shifted

    score_map = ScoreMap.from_shape(seal_impression_layout.shape)

    for char_slot_idx, (char_box, char_glyph) \
            in enumerate(zip(text_line.char_boxes, text_line.char_glyphs)):
        # Get char slot to be filled.
        if char_slot_idx >= len(seal_impression_layout.char_slots):
            break
        char_slot = seal_impression_layout.char_slots[char_slot_idx]

        # Extract char-level score map.
        box = char_box.box
        char_score_map = ScoreMap.from_shape((text_line.box.height, box.width))
        if char_glyph.score_map:
            char_glyph_score_map = char_glyph.score_map
            if char_glyph_score_map.shape != box.shape:
                char_glyph_score_map = char_glyph_score_map.to_resized_score_map(
                    resized_height=box.height,
                    resized_width=box.width,
                    cv_resize_interpolation=text_line.cv_resize_interpolation,
                )
            char_score_map.mat[box.up:box.down + 1] = char_glyph_score_map.mat
        else:
            # LCD, fallback to mask.
            char_glyph_mask = char_glyph.get_glyph_mask(
                box=box,
                cv_resize_interpolation=text_line.cv_resize_interpolation,
            )
            char_score_map.mat[box.up:box.down + 1] = char_glyph_mask.mat.astype(np.float32)

        point_up = Point(y=0, x=char_score_map.width // 2)

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
        dst_down = char_slot.point_up.y + (rotated_char_score_map.height - 1 - rotated_point_up.y)
        dst_left = char_slot.point_up.x - rotated_point_up.x
        dst_right = char_slot.point_up.x + (rotated_char_score_map.width - 1 - rotated_point_up.x)

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

    # Adjust alpha.
    score_map_max = score_map.mat.max()
    score_map.mat = score_map.mat * seal_impression_layout.alpha / score_map_max

    return score_map
