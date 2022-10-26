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
from typing import Optional, Any, Callable, List, Sequence
import itertools

import attrs
import numpy as np
from numpy.random import Generator as RandomGenerator
import cv2 as cv
import freetype

from vkit.utility import sample_cv_resize_interpolation
from vkit.element import Image, Box, Mask, ScoreMap
from vkit.engine.interface import (
    NoneTypeEngineInitConfig,
    NoneTypeEngineInitResource,
    Engine,
    EngineExecutorFactory,
)
from .type import (
    CharBox,
    FontEngineRunConfigGlyphSequence,
    FontEngineRunConfigStyle,
    FontEngineRunConfig,
    CharGlyph,
    TextLine,
)


def estimate_font_size(config: FontEngineRunConfig):
    style = config.style

    # Estimate the font size based on height or width.
    if config.glyph_sequence == FontEngineRunConfigGlyphSequence.HORI_DEFAULT:
        font_size = round(config.height * style.font_size_ratio)
    elif config.glyph_sequence == FontEngineRunConfigGlyphSequence.VERT_DEFAULT:
        font_size = round(config.width * style.font_size_ratio)
    else:
        raise NotImplementedError()

    font_size = int(np.clip(font_size, style.font_size_min, style.font_size_max))
    return font_size


def load_freetype_font_face(
    run_config: FontEngineRunConfig,
    lcd_compression_factor: Optional[int] = None,
):
    font_variant = run_config.font_variant
    if font_variant.is_ttc:
        assert font_variant.ttc_font_index is not None
        font_face = freetype.Face(str(font_variant.font_file), index=font_variant.ttc_font_index)
    else:
        font_face = freetype.Face(str(font_variant.font_file))

    font_size = estimate_font_size(run_config)

    # 1. "The nominal width, in 26.6 fractional points" and "The 26.6 fixed float format
    #    used to define fractional pixel coordinates. Here, 1 unit = 1/64 pixel", hence
    #    we need to multiple 64 here.
    # 2. `height` defaults to 0 and "If either the character width or
    #    height is zero, it is set equal to the other value",
    #    hence no need to set the `height` parameter.
    # 3. `vres` for vertical resolution, and defaults to 72.
    #    Since "pixel_size = point_size * resolution / 72", setting resolution to 72 equalize
    #    point_size and pixel_size. Since "If either the horizontal or vertical resolution is zero,
    #    it is set equal to the other value" and `vres` defaults to 72, we don't need to set the
    #    `vres` parameter.
    # 4. `hres` for horizontal resolution, and defaults to 72. If `lcd_compression_factor` is set,
    #    we need to cancel the sub-pixel rendering effect, by scaling up the same factor to `hres`.
    hres = 72
    if lcd_compression_factor is not None:
        hres *= lcd_compression_factor
    font_face.set_char_size(width=font_size * 64, hres=hres)

    return font_face


def build_freetype_font_face_lcd_hc_matrix(lcd_compression_factor: int):
    # "hc" stands for "horizontal compression".
    return freetype.Matrix(
        int((1 / lcd_compression_factor) * 0x10000),
        int(0.0 * 0x10000),
        int(0.0 * 0x10000),
        int(1.0 * 0x10000),
    )


def trim_char_np_image_vert(np_image: np.ndarray):
    # (H, W) or (H, W, 3)
    np_vert_max: np.ndarray = np.amax(np_image, axis=1)
    if np_vert_max.ndim == 2:
        np_vert_max = np.amax(np_vert_max, axis=1)
    assert np_vert_max.ndim == 1

    np_vert_nonzero = np.nonzero(np_vert_max)[0]
    if len(np_vert_nonzero) == 0:
        raise RuntimeError('trim_char_np_image_vert: empty np_image.')

    up = int(np_vert_nonzero[0])
    down = int(np_vert_nonzero[-1])

    height = np_image.shape[0]
    return np_image[up:down + 1], up, height - 1 - down


def trim_char_np_image_hori(np_image: np.ndarray):
    # (H, W) or (H, W, 3)
    np_hori_max: np.ndarray = np.amax(np_image, axis=0)
    if np_hori_max.ndim == 2:
        np_hori_max = np.amax(np_hori_max, axis=1)
    assert np_hori_max.ndim == 1

    np_hori_nonzero = np.nonzero(np_hori_max)[0]
    if len(np_hori_nonzero) == 0:
        raise RuntimeError('trim_char_np_image_hori: empty np_image.')

    left = int(np_hori_nonzero[0])
    right = int(np_hori_nonzero[-1])

    width = np_image.shape[1]
    return np_image[:, left:right + 1], left, width - 1 - right


def build_char_glyph(
    config: FontEngineRunConfig,
    char: str,
    glyph: Any,
    np_image: np.ndarray,
):
    assert not char.isspace()

    # References:
    # https://freetype.org/freetype2/docs/tutorial/step2.html
    # https://freetype-py.readthedocs.io/en/latest/glyph_slot.html

    # "the distance from the baseline to the top-most glyph scanline"
    ascent = glyph.bitmap_top

    # "It is always zero for horizontal layouts, and positive for vertical layouts."
    assert glyph.advance.y == 0

    # Trim vertically to get pad_up & pad_down.
    np_image, pad_up, pad_down = trim_char_np_image_vert(np_image)
    ascent -= pad_up

    # "bitmap’s left bearing"
    # NOTE: `bitmap_left` could be negative, we simply ignore the negative value.
    pad_left = max(0, glyph.bitmap_left)

    # "It is always zero for horizontal layouts, and zero for vertical ones."
    assert glyph.advance.x > 0
    width = np_image.shape[1]
    # Aforementioned, 1 unit = 1/64 pixel.
    pad_right = round(glyph.advance.x / 64) - pad_left - width
    # Apply defensive clipping.
    pad_right = max(0, pad_right)

    # Trim horizontally to increase pad_left & pad_right.
    np_image, pad_left_inc, pad_right_inc = trim_char_np_image_hori(np_image)
    pad_left += pad_left_inc
    pad_right += pad_right_inc

    score_map = None
    if np_image.ndim == 2:
        # Apply gamma correction.
        np_alpha = np.power(
            np_image.astype(np.float32) / 255.0,
            config.style.glyph_color_gamma,
        )
        score_map = ScoreMap(mat=np_alpha)

    # Reference statistics.
    font_variant = config.font_variant
    tag_to_font_glyph_info = font_variant.font_glyph_info_collection.tag_to_font_glyph_info

    assert char in font_variant.char_to_tags
    tags = font_variant.char_to_tags[char]

    font_glyph_info = None
    for tag in tags:
        assert tag in tag_to_font_glyph_info
        cur_font_glyph_info = tag_to_font_glyph_info[tag]
        if font_glyph_info is None:
            font_glyph_info = cur_font_glyph_info
        else:
            assert font_glyph_info == cur_font_glyph_info

    assert font_glyph_info is not None

    font_size = estimate_font_size(config)
    ref_ascent_plus_pad_up = round(
        font_glyph_info.ascent_plus_pad_up_min_to_font_size_ratio * font_size
    )
    ref_char_height = round(font_glyph_info.height_min_to_font_size_ratio * font_size)
    ref_char_width = round(font_glyph_info.width_min_to_font_size_ratio * font_size)

    return CharGlyph(
        char=char,
        image=Image(mat=np_image),
        score_map=score_map,
        ascent=ascent,
        pad_up=pad_up,
        pad_down=pad_down,
        pad_left=pad_left,
        pad_right=pad_right,
        ref_ascent_plus_pad_up=ref_ascent_plus_pad_up,
        ref_char_height=ref_char_height,
        ref_char_width=ref_char_width,
    )


def render_char_glyphs_from_text(
    run_config: FontEngineRunConfig,
    font_face: freetype.Face,
    func_render_char_glyph: Callable[[FontEngineRunConfig, freetype.Face, str], CharGlyph],
    chars: Sequence[str],
):
    char_glyphs: List[CharGlyph] = []
    prev_num_spaces_for_char_glyphs: List[int] = []
    num_spaces = 0
    for idx, char in enumerate(chars):
        if char.isspace():
            num_spaces += 1
            continue

        char_glyphs.append(func_render_char_glyph(run_config, font_face, char))

        if idx == 0 and num_spaces > 0:
            raise RuntimeError('Leading space(s) detected.')
        prev_num_spaces_for_char_glyphs.append(num_spaces)
        num_spaces = 0

    if num_spaces > 0:
        raise RuntimeError('Trailing space(s) detected.')

    return char_glyphs, prev_num_spaces_for_char_glyphs


def get_kerning_limits_hori_default(
    char_glyphs: Sequence[CharGlyph],
    prev_num_spaces_for_char_glyphs: Sequence[int],
):
    assert char_glyphs
    ascent_max = max(char_glyph.ascent for char_glyph in char_glyphs)

    kerning_limits: List[int] = []

    prev_glyph_mask = None
    prev_np_glyph_mask = None
    prev_glyph_mask_up = None
    prev_glyph_mask_down = None
    for char_glyph, prev_num_spaces in zip(char_glyphs, prev_num_spaces_for_char_glyphs):
        glyph_mask = char_glyph.get_glyph_mask()
        np_glyph_mask = glyph_mask.mat
        glyph_mask_up = ascent_max - char_glyph.ascent
        glyph_mask_down = glyph_mask_up + np_glyph_mask.shape[0] - 1

        if prev_num_spaces == 0 and prev_np_glyph_mask is not None:
            assert prev_glyph_mask is not None
            assert prev_glyph_mask_up is not None
            assert prev_glyph_mask_down is not None

            overlap_up = max(prev_glyph_mask_up, glyph_mask_up)
            overlap_down = min(prev_glyph_mask_down, glyph_mask_down)
            if overlap_up <= overlap_down:
                overlap_prev_np_glyph_mask = \
                    prev_np_glyph_mask[overlap_up - prev_glyph_mask_up:
                                       overlap_down - prev_glyph_mask_up + 1]
                overlap_np_glyph_mask = \
                    np_glyph_mask[overlap_up - glyph_mask_up:
                                  overlap_down - glyph_mask_up + 1]

                kerning_limit = 1
                while kerning_limit < prev_glyph_mask.width / 2 \
                        and kerning_limit < glyph_mask.width / 2:
                    prev_np_glyph_mask_tail = overlap_prev_np_glyph_mask[:, -kerning_limit:]
                    np_glyph_mask_head = overlap_np_glyph_mask[:, :kerning_limit]
                    if (prev_np_glyph_mask_tail & np_glyph_mask_head).any():
                        # Intersection detected.
                        kerning_limit -= 1
                        break
                    kerning_limit += 1

                kerning_limits.append(kerning_limit)

            else:
                # Not overlapped.
                kerning_limits.append(0)

        else:
            # Isolated by word space or being the first glyph, skip.
            kerning_limits.append(0)

        prev_glyph_mask = glyph_mask
        prev_np_glyph_mask = np_glyph_mask
        prev_glyph_mask_up = glyph_mask_up
        prev_glyph_mask_down = glyph_mask_down

    return kerning_limits


def render_char_glyphs_in_text_line(
    style: FontEngineRunConfigStyle,
    text_line_height: int,
    text_line_width: int,
    char_glyphs: Sequence[CharGlyph],
    char_boxes: Sequence[CharBox],
):
    # Genrate text line image.
    np_image = np.full((text_line_height, text_line_width, 3), 255, dtype=np.uint8)
    np_mask = np.zeros((text_line_height, text_line_width), dtype=np.uint8)
    score_map = None

    if char_glyphs[0].image.mat.ndim == 2:
        score_map = ScoreMap.from_shape((text_line_height, text_line_width))

        # Default or monochrome.
        for char_glyph, char_box in zip(char_glyphs, char_boxes):
            assert char_glyph.score_map

            char_glyph_mask = char_glyph.get_glyph_mask(box=char_box.box)

            # Fill color based on RGBA & alpha.
            np_char_image = np.full(
                (char_glyph.height, char_glyph.width, 4),
                (*style.glyph_color, 0),
                dtype=np.uint8,
            )
            np_char_image[:, :, 3] = (char_glyph.score_map.mat * 255).astype(np.uint8)

            # To RGB.
            np_char_image = cv.cvtColor(np_char_image, cv.COLOR_RGBA2RGB)

            # Paste to text line.
            char_glyph_mask.fill_np_array(np_image, np_char_image)
            char_glyph_mask.fill_np_array(np_mask, 1)
            char_box.box.fill_score_map(
                score_map,
                char_glyph.score_map,
                keep_max_value=True,
            )

    elif char_glyphs[0].image.mat.ndim == 3:
        # LCD.
        for char_glyph, char_box in zip(char_glyphs, char_boxes):
            char_glyph_mask = char_glyph.get_glyph_mask(box=char_box.box)

            # NOTE: the `glyph_color` option is ignored in LCD mode.
            # Gamma correction.
            np_char_image = np.power(
                char_glyph.image.mat / 255.0,
                style.glyph_color_gamma,
            )
            np_char_image = ((1 - np_char_image) * 255).astype(np.uint8)  # type: ignore

            # Paste to text line.
            char_glyph_mask.fill_np_array(np_image, np_char_image)
            char_glyph_mask.fill_np_array(np_mask, 1)

    else:
        raise NotImplementedError()

    return (
        Image(mat=np_image),
        Mask(mat=np_mask),
        score_map,
        char_boxes,
    )


def place_char_glyphs_in_text_line_hori_default(
    run_config: FontEngineRunConfig,
    char_glyphs: Sequence[CharGlyph],
    prev_num_spaces_for_char_glyphs: Sequence[int],
    kerning_limits: Sequence[int],
    rng: RandomGenerator,
):
    style = run_config.style

    assert char_glyphs
    char_widths_avg = np.mean([char_glyph.width for char_glyph in char_glyphs])

    char_space_min = char_widths_avg * style.char_space_min
    char_space_max = char_widths_avg * style.char_space_max
    char_space_mean = char_widths_avg * style.char_space_mean
    char_space_std = char_widths_avg * style.char_space_std

    word_space_min = char_widths_avg * style.word_space_min
    word_space_max = char_widths_avg * style.word_space_max
    word_space_mean = char_widths_avg * style.word_space_mean
    word_space_std = char_widths_avg * style.word_space_std

    ascent_plus_pad_up_max = max(
        itertools.chain.from_iterable(
            (char_glyph.ascent + char_glyph.pad_up, char_glyph.ref_ascent_plus_pad_up)
            for char_glyph in char_glyphs
        )
    )

    text_line_height = max(char_glyph.ref_char_height for char_glyph in char_glyphs)

    char_boxes: List[CharBox] = []
    hori_offset = 0
    for char_idx, (char_glyph, prev_num_spaces, kerning_limit) in enumerate(
        zip(
            char_glyphs,
            prev_num_spaces_for_char_glyphs,
            kerning_limits,
        )
    ):
        # "Stick" chars together.
        hori_offset -= kerning_limit

        # Shift by space.
        if prev_num_spaces > 0:
            # Random word space(s).
            space = 0
            for _ in range(prev_num_spaces):
                space += round(
                    np.clip(
                        rng.normal(loc=word_space_mean, scale=word_space_std),
                        word_space_min,
                        word_space_max,
                    )  # type: ignore
                )

        else:
            # Random char space.
            space = round(
                np.clip(
                    rng.normal(loc=char_space_mean, scale=char_space_std),
                    char_space_min,
                    char_space_max,
                )  # type: ignore
            )

        hori_offset += space

        # Place char box.
        up = ascent_plus_pad_up_max - char_glyph.ascent
        down = up + char_glyph.height - 1

        left = hori_offset + char_glyph.pad_left
        if char_idx == 0:
            # Ignore the leading padding.
            left = 0
        right = left + char_glyph.width - 1

        assert not char_glyph.char.isspace()
        char_boxes.append(
            CharBox(
                char=char_glyph.char,
                box=Box(
                    up=up,
                    down=down,
                    left=left,
                    right=right,
                ),
            )
        )

        # Update the height of text line.
        text_line_height = max(text_line_height, down + 1 + char_glyph.pad_down)

        # Move offset.
        hori_offset = right + 1
        if char_idx < len(char_glyphs) - 1:
            hori_offset += char_glyph.pad_right

    text_line_width = hori_offset

    return render_char_glyphs_in_text_line(
        style=style,
        text_line_height=text_line_height,
        text_line_width=text_line_width,
        char_glyphs=char_glyphs,
        char_boxes=char_boxes,
    )


def place_char_glyphs_in_text_line_vert_default(
    run_config: FontEngineRunConfig,
    char_glyphs: Sequence[CharGlyph],
    prev_num_spaces_for_char_glyphs: Sequence[int],
    rng: RandomGenerator,
):
    style = run_config.style

    assert char_glyphs
    char_widths_avg = np.mean([char_glyph.width for char_glyph in char_glyphs])

    char_space_min = char_widths_avg * style.char_space_min
    char_space_max = char_widths_avg * style.char_space_max
    char_space_mean = char_widths_avg * style.char_space_mean
    char_space_std = char_widths_avg * style.char_space_std

    word_space_min = char_widths_avg * style.word_space_min
    word_space_max = char_widths_avg * style.word_space_max
    word_space_mean = char_widths_avg * style.word_space_mean
    word_space_std = char_widths_avg * style.word_space_std

    text_line_width = max(
        itertools.chain.from_iterable((
            char_glyph.pad_left + char_glyph.width + char_glyph.pad_right,
            char_glyph.ref_char_width,
        ) for char_glyph in char_glyphs)
    )

    text_line_width_mid = text_line_width // 2

    char_boxes: List[CharBox] = []
    vert_offset = 0
    for char_idx, (char_glyph, prev_num_spaces) in enumerate(
        zip(char_glyphs, prev_num_spaces_for_char_glyphs)
    ):
        # Shift by space.
        if prev_num_spaces > 0:
            # Random word space(s).
            space = 0
            for _ in range(prev_num_spaces):
                space += round(
                    np.clip(
                        rng.normal(loc=word_space_mean, scale=word_space_std),
                        word_space_min,
                        word_space_max,
                    )  # type: ignore
                )

        else:
            # Random char space.
            space = round(
                np.clip(
                    rng.normal(loc=char_space_mean, scale=char_space_std),
                    char_space_min,
                    char_space_max,
                )  # type: ignore
            )

        vert_offset += space

        # Place char box.
        up = vert_offset + char_glyph.pad_up
        if char_idx == 0:
            # Ignore the leading padding.
            up = 0

        down = up + char_glyph.height - 1

        # Vertical align in middle.
        left = text_line_width_mid - char_glyph.width // 2
        right = left + char_glyph.width - 1

        assert not char_glyph.char.isspace()
        char_boxes.append(
            CharBox(
                char=char_glyph.char,
                box=Box(
                    up=up,
                    down=down,
                    left=left,
                    right=right,
                ),
            )
        )

        # Move offset.
        vert_offset = down + 1
        if char_idx < len(char_glyphs) - 1:
            vert_offset += char_glyph.pad_down

    text_line_height = vert_offset

    return render_char_glyphs_in_text_line(
        style=style,
        text_line_height=text_line_height,
        text_line_width=text_line_width,
        char_glyphs=char_glyphs,
        char_boxes=char_boxes,
    )


def resize_and_trim_text_line_hori_default(
    run_config: FontEngineRunConfig,
    cv_resize_interpolation_enlarge: int,
    cv_resize_interpolation_shrink: int,
    image: Image,
    mask: Mask,
    score_map: Optional[ScoreMap],
    char_boxes: Sequence[CharBox],
    char_glyphs: Sequence[CharGlyph],
):
    # Resize if image height too small or too large.
    is_too_small = (image.height / run_config.height < 0.8)
    is_too_large = (image.height > run_config.height)

    cv_resize_interpolation = cv_resize_interpolation_enlarge
    if is_too_large:
        cv_resize_interpolation = cv_resize_interpolation_shrink

    if is_too_small or is_too_large:
        resized_image = image.to_resized_image(
            resized_height=run_config.height,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_mask = mask.to_resized_mask(
            resized_height=run_config.height,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_char_boxes = [
            char_box.to_conducted_resized_char_box(
                shapable_or_shape=image,
                resized_height=run_config.height,
            ) for char_box in char_boxes
        ]

        image = resized_image
        mask = resized_mask
        char_boxes = resized_char_boxes

        if score_map:
            score_map = score_map.to_resized_score_map(
                resized_height=run_config.height,
                cv_resize_interpolation=cv_resize_interpolation,
            )

    # Pad vertically.
    if image.height != run_config.height:
        pad_vert = run_config.height - image.height
        assert pad_vert > 0
        pad_up = pad_vert // 2
        pad_down = pad_vert - pad_up

        np_image = np.full((run_config.height, image.width, 3), 255, dtype=np.uint8)
        np_image[pad_up:-pad_down] = image.mat
        image.assign_mat(np_image)

        np_mask = np.zeros((run_config.height, image.width), dtype=np.uint8)
        np_mask[pad_up:-pad_down] = mask.mat
        mask.assign_mat(np_mask)

        padded_char_boxes = []
        for char_box in char_boxes:
            box = attrs.evolve(
                char_box.box,
                up=char_box.up + pad_up,
                down=char_box.down + pad_up,
            )
            padded_char_boxes.append(attrs.evolve(char_box, box=box))
        char_boxes = padded_char_boxes

        if score_map:
            padded_score_map = ScoreMap.from_shape((run_config.height, image.width))
            with padded_score_map.writable_context:
                padded_score_map.mat[pad_up:-pad_down] = score_map.mat
            score_map = padded_score_map

    # Trim.
    if image.width > run_config.width:
        last_char_box_idx = len(char_boxes) - 1
        while last_char_box_idx >= 0 and char_boxes[last_char_box_idx].right >= run_config.width:
            last_char_box_idx -= 1

        if last_char_box_idx == len(char_boxes) - 1:
            # Corner case: char_boxes[-1].right < config.width but mage.width > config.width.
            # This is caused by glyph padding. The solution is to drop this char.
            last_char_box_idx -= 1

        if last_char_box_idx < 0 or char_boxes[last_char_box_idx].right >= run_config.width:
            # Cannot trim.
            return None, None, None, None, -1

        last_char_box = char_boxes[last_char_box_idx]
        last_char_box_right = last_char_box.right

        # Clean up residual pixels.
        first_trimed_char_box = char_boxes[last_char_box_idx + 1]
        if first_trimed_char_box.left <= last_char_box_right:
            first_trimed_char_glyph = char_glyphs[last_char_box_idx + 1]

            first_trimed_char_glyph_mask = first_trimed_char_glyph.get_glyph_mask(
                box=first_trimed_char_box.box,
                enable_resize=True,
                cv_resize_interpolation=cv_resize_interpolation,
            )
            first_trimed_char_glyph_mask.fill_image(image, (255, 255, 255))
            first_trimed_char_glyph_mask.fill_mask(mask, 0)

            if first_trimed_char_glyph.score_map:
                assert score_map

                first_trimed_char_score_map = first_trimed_char_glyph.score_map
                if first_trimed_char_score_map.shape != first_trimed_char_box.shape:
                    first_trimed_char_score_map = first_trimed_char_score_map.to_resized_score_map(
                        resized_height=first_trimed_char_box.height,
                        resized_width=first_trimed_char_box.width,
                        cv_resize_interpolation=cv_resize_interpolation,
                    )

                last_char_score_map = char_glyphs[last_char_box_idx].score_map
                assert last_char_score_map
                if last_char_score_map.shape != last_char_box.shape:
                    last_char_score_map = last_char_score_map.to_resized_score_map(
                        resized_height=last_char_box.height,
                        resized_width=last_char_box.width,
                        cv_resize_interpolation=cv_resize_interpolation,
                    )

                first_trimed_char_box.box.fill_score_map(score_map, 0)
                last_char_box.box.fill_score_map(
                    score_map,
                    last_char_score_map,
                    keep_max_value=True,
                )

        char_boxes = char_boxes[:last_char_box_idx + 1]
        image.assign_mat(image.mat[:, :last_char_box_right + 1])
        mask.assign_mat(mask.mat[:, :last_char_box_right + 1])

        if score_map:
            score_map.assign_mat(score_map.mat[:, :last_char_box_right + 1])

    return image, mask, score_map, char_boxes, cv_resize_interpolation


def resize_and_trim_text_line_vert_default(
    run_config: FontEngineRunConfig,
    cv_resize_interpolation_enlarge: int,
    cv_resize_interpolation_shrink: int,
    image: Image,
    mask: Mask,
    score_map: Optional[ScoreMap],
    char_boxes: Sequence[CharBox],
):
    # Resize if image width too small or too large.
    is_too_small = (image.width / run_config.width < 0.8)
    is_too_large = (image.width > run_config.width)

    cv_resize_interpolation = cv_resize_interpolation_enlarge
    if is_too_large:
        cv_resize_interpolation = cv_resize_interpolation_shrink

    if is_too_small or is_too_large:
        resized_image = image.to_resized_image(
            resized_width=run_config.width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_mask = mask.to_resized_mask(
            resized_width=run_config.width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_char_boxes = [
            char_box.to_conducted_resized_char_box(
                shapable_or_shape=image,
                resized_width=run_config.width,
            ) for char_box in char_boxes
        ]

        image = resized_image
        mask = resized_mask
        char_boxes = resized_char_boxes

        if score_map:
            score_map = score_map.to_resized_score_map(
                resized_width=run_config.width,
                cv_resize_interpolation=cv_resize_interpolation,
            )

    # Pad horizontally.
    if image.width != run_config.width:
        pad_hori = run_config.width - image.width
        assert pad_hori > 0
        pad_left = pad_hori // 2
        pad_right = pad_hori - pad_left

        np_image = np.full((image.height, run_config.width, 3), 255, dtype=np.uint8)
        np_image[:, pad_left:-pad_right] = image.mat
        image.assign_mat(np_image)

        np_mask = np.zeros((image.height, run_config.width), dtype=np.uint8)
        np_mask[:, pad_left:-pad_right] = mask.mat
        mask.assign_mat(np_mask)

        padded_char_boxes = []
        for char_box in char_boxes:
            box = attrs.evolve(
                char_box.box,
                left=char_box.left + pad_left,
                right=char_box.right + pad_right,
            )
            padded_char_boxes.append(attrs.evolve(char_box, box=box))
        char_boxes = padded_char_boxes

        if score_map:
            padded_score_map = ScoreMap.from_shape((image.height, run_config.width))
            padded_score_map.mat[:, pad_left:-pad_right] = score_map.mat
            score_map = padded_score_map

    # Trim.
    if image.height > run_config.height:
        last_char_box_idx = len(char_boxes) - 1
        while last_char_box_idx >= 0 and char_boxes[last_char_box_idx].down >= run_config.height:
            last_char_box_idx -= 1

        if last_char_box_idx == len(char_boxes) - 1:
            last_char_box_idx -= 1

        if last_char_box_idx < 0 or char_boxes[last_char_box_idx].down >= run_config.height:
            # Cannot trim.
            return None, None, None, None, -1

        last_char_box_down = char_boxes[last_char_box_idx].down
        char_boxes = char_boxes[:last_char_box_idx + 1]
        image.assign_mat(image.mat[:last_char_box_down + 1])
        mask.assign_mat(mask.mat[:last_char_box_down + 1])

        if score_map:
            score_map.assign_mat(score_map.mat[:last_char_box_down + 1])

    return image, mask, score_map, char_boxes, cv_resize_interpolation


def render_text_line_meta(
    run_config: FontEngineRunConfig,
    font_face: freetype.Face,
    func_render_char_glyph: Callable[[FontEngineRunConfig, freetype.Face, str], CharGlyph],
    rng: RandomGenerator,
    cv_resize_interpolation_enlarge: int = cv.INTER_CUBIC,
    cv_resize_interpolation_shrink: int = cv.INTER_AREA,
):
    (
        char_glyphs,
        prev_num_spaces_for_char_glyphs,
    ) = render_char_glyphs_from_text(
        run_config=run_config,
        font_face=font_face,
        func_render_char_glyph=func_render_char_glyph,
        chars=run_config.chars,
    )
    if not char_glyphs:
        return None

    if run_config.glyph_sequence == FontEngineRunConfigGlyphSequence.HORI_DEFAULT:
        kerning_limits = get_kerning_limits_hori_default(
            char_glyphs,
            prev_num_spaces_for_char_glyphs,
        )
        (
            image,
            mask,
            score_map,
            char_boxes,
        ) = place_char_glyphs_in_text_line_hori_default(
            run_config=run_config,
            char_glyphs=char_glyphs,
            prev_num_spaces_for_char_glyphs=prev_num_spaces_for_char_glyphs,
            kerning_limits=kerning_limits,
            rng=rng,
        )
        (
            image,
            mask,
            score_map,
            char_boxes,
            cv_resize_interpolation,
        ) = resize_and_trim_text_line_hori_default(
            run_config=run_config,
            cv_resize_interpolation_enlarge=cv_resize_interpolation_enlarge,
            cv_resize_interpolation_shrink=cv_resize_interpolation_shrink,
            image=image,
            mask=mask,
            score_map=score_map,
            char_boxes=char_boxes,
            char_glyphs=char_glyphs,
        )
        is_hori = True

    elif run_config.glyph_sequence == FontEngineRunConfigGlyphSequence.VERT_DEFAULT:
        # NOTE: No kerning limit detection for VERT_DEFAULT mode.
        (
            image,
            mask,
            score_map,
            char_boxes,
        ) = place_char_glyphs_in_text_line_vert_default(
            run_config=run_config,
            char_glyphs=char_glyphs,
            prev_num_spaces_for_char_glyphs=prev_num_spaces_for_char_glyphs,
            rng=rng,
        )
        (
            image,
            mask,
            score_map,
            char_boxes,
            cv_resize_interpolation,
        ) = resize_and_trim_text_line_vert_default(
            run_config=run_config,
            cv_resize_interpolation_enlarge=cv_resize_interpolation_enlarge,
            cv_resize_interpolation_shrink=cv_resize_interpolation_shrink,
            image=image,
            mask=mask,
            score_map=score_map,
            char_boxes=char_boxes,
        )
        is_hori = False

    else:
        raise NotImplementedError()

    if image is None:
        return None
    else:
        assert mask is not None
        assert char_boxes is not None

        char_idx = 0
        non_space_count = 0
        while char_idx < len(run_config.chars) and non_space_count < len(char_boxes):
            if not run_config.chars[char_idx].isspace():
                non_space_count += 1
            char_idx += 1
        assert non_space_count == len(char_boxes)

        box = Box.from_shapable(image)
        image = image.to_box_attached(box)
        mask = mask.to_box_attached(box)
        if score_map:
            score_map = score_map.to_box_attached(box)

        return TextLine(
            image=image,
            mask=mask,
            score_map=score_map,
            char_boxes=char_boxes,
            char_glyphs=char_glyphs[:len(char_boxes)],
            cv_resize_interpolation=cv_resize_interpolation,
            font_size=estimate_font_size(run_config),
            style=run_config.style,
            text=''.join(run_config.chars[:char_idx]),
            is_hori=is_hori,
            font_variant=run_config.font_variant if run_config.return_font_variant else None,
        )


class FontFreetypeDefaultEngine(
    Engine[
        NoneTypeEngineInitConfig,
        NoneTypeEngineInitResource,
        FontEngineRunConfig,
        Optional[TextLine],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'freetype_default'

    @classmethod
    def render_char_glyph(
        cls,
        run_config: FontEngineRunConfig,
        font_face: freetype.Face,
        char: str,
    ):
        load_char_flags = freetype.FT_LOAD_RENDER  # type: ignore
        if run_config.style.freetype_force_autohint:
            load_char_flags |= freetype.FT_LOAD_FORCE_AUTOHINT  # type: ignore
        font_face.load_char(char, load_char_flags)

        glyph = font_face.glyph
        bitmap = glyph.bitmap

        height = bitmap.rows
        width = bitmap.width
        assert width == bitmap.pitch

        # (H, W), [0, 255]
        np_image = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(height, width)

        return build_char_glyph(run_config, char, glyph, np_image)

    def run(self, run_config: FontEngineRunConfig, rng: RandomGenerator) -> Optional[TextLine]:
        font_face = load_freetype_font_face(run_config)
        return render_text_line_meta(
            run_config=run_config,
            font_face=font_face,
            func_render_char_glyph=self.render_char_glyph,
            rng=rng,
            cv_resize_interpolation_enlarge=sample_cv_resize_interpolation(rng),
            cv_resize_interpolation_shrink=sample_cv_resize_interpolation(
                rng,
                include_cv_inter_area=True,
            ),
        )


font_freetype_default_engine_executor_factory = EngineExecutorFactory(FontFreetypeDefaultEngine)


class FontFreetypeLcdEngine(
    Engine[
        NoneTypeEngineInitConfig,
        NoneTypeEngineInitResource,
        FontEngineRunConfig,
        Optional[TextLine],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'freetype_lcd'

    @classmethod
    def render_char_glyph(
        cls,
        run_config: FontEngineRunConfig,
        font_face: freetype.Face,
        lcd_hc_matrix: freetype.Matrix,
        char: str,
    ):
        load_char_flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_LCD  # type: ignore
        if run_config.style.freetype_force_autohint:
            load_char_flags |= freetype.FT_LOAD_FORCE_AUTOHINT  # type: ignore
        font_face.set_transform(lcd_hc_matrix, freetype.Vector(0, 0))
        font_face.load_char(char, load_char_flags)

        glyph = font_face.glyph
        bitmap = glyph.bitmap

        height = bitmap.rows
        pitch = bitmap.pitch
        flatten_width = bitmap.width
        width = flatten_width // 3

        # (H, W, 3), [0, 255]
        np_image = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(height, pitch)
        np_image = np_image[:, :width * 3].reshape(height, width, 3)

        return build_char_glyph(run_config, char, glyph, np_image)

    @classmethod
    def bind_render_char_glyph(cls, lcd_hc_matrix: freetype.Matrix):
        return lambda config, font_face, char: cls.render_char_glyph(
            config,
            font_face,
            lcd_hc_matrix,
            char,
        )

    def run(self, run_config: FontEngineRunConfig, rng: RandomGenerator) -> Optional[TextLine]:
        lcd_compression_factor = 10
        font_face = load_freetype_font_face(
            run_config,
            lcd_compression_factor=lcd_compression_factor,
        )
        lcd_hc_matrix = build_freetype_font_face_lcd_hc_matrix(lcd_compression_factor)
        return render_text_line_meta(
            run_config=run_config,
            font_face=font_face,
            func_render_char_glyph=self.bind_render_char_glyph(lcd_hc_matrix),
            rng=rng,
            cv_resize_interpolation_enlarge=sample_cv_resize_interpolation(rng),
            cv_resize_interpolation_shrink=sample_cv_resize_interpolation(
                rng,
                include_cv_inter_area=True,
            ),
        )


font_freetype_lcd_engine_executor_factory = EngineExecutorFactory(FontFreetypeLcdEngine)


class FontFreetypeMonochromeEngine(
    Engine[
        NoneTypeEngineInitConfig,
        NoneTypeEngineInitResource,
        FontEngineRunConfig,
        Optional[TextLine],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'freetype_monochrome'

    @classmethod
    def render_char_glyph(
        cls,
        run_config: FontEngineRunConfig,
        font_face: freetype.Face,
        char: str,
    ):
        load_char_flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO  # type: ignore
        if run_config.style.freetype_force_autohint:
            load_char_flags |= freetype.FT_LOAD_FORCE_AUTOHINT  # type: ignore
        font_face.load_char(char, load_char_flags)

        glyph = font_face.glyph
        bitmap = glyph.bitmap

        height = bitmap.rows
        width = bitmap.width
        pitch = bitmap.pitch

        # Performance optimization.
        bitmap_buffer = bitmap.buffer

        data = []
        for row_idx in range(height):
            row = []
            for pitch_idx in range(pitch):
                byte = bitmap_buffer[row_idx * pitch + pitch_idx]
                bits = []
                for _ in range(8):
                    bits.append(int((byte & 1) == 1))
                    byte = byte >> 1
                row.extend(bit * 255 for bit in reversed(bits))
            data.append(row[:width])

        np_image = np.asarray(data, dtype=np.uint8)
        assert np_image.shape == (height, width)

        return build_char_glyph(run_config, char, glyph, np_image)

    def run(self, run_config: FontEngineRunConfig, rng: RandomGenerator) -> Optional[TextLine]:
        font_face = load_freetype_font_face(run_config)
        return render_text_line_meta(
            run_config=run_config,
            font_face=font_face,
            func_render_char_glyph=self.render_char_glyph,
            rng=rng,
            cv_resize_interpolation_enlarge=cv.INTER_NEAREST_EXACT,
            cv_resize_interpolation_shrink=cv.INTER_NEAREST_EXACT,
        )


font_freetype_monochrome_engine_executor_factory = EngineExecutorFactory(
    FontFreetypeMonochromeEngine
)
