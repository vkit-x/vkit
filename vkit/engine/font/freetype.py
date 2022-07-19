from typing import Optional, Any, Callable, List, Sequence

import attrs
import numpy as np
from numpy.random import Generator as RandomGenerator
import cv2 as cv
import freetype

from vkit.utility import sample_cv_resize_interpolation
from vkit.element import Image, CharBox, Box, Mask, ScoreMap
from vkit.engine.interface import (
    Engine,
    NoneTypeEngineConfig,
    NoneTypeEngineResource,
)
from .type import (
    FontEngineRunConfigGlyphSequence,
    FontEngineRunConfigStyle,
    FontEngineRunConfig,
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


def get_ref_ascent_plus_pad_up_min(config: FontEngineRunConfig):
    font_size = estimate_font_size(config)
    ratio = config.font_variant.ascent_plus_pad_up_min_to_font_size_ratio
    return round(ratio * font_size)


def get_ref_char_height(config: FontEngineRunConfig):
    font_size = estimate_font_size(config)
    ratio = config.font_variant.height_min_to_font_size_ratio
    return round(ratio * font_size)


def get_ref_char_width(config: FontEngineRunConfig):
    font_size = estimate_font_size(config)
    ratio = config.font_variant.width_min_to_font_size_ratio
    return round(ratio * font_size)


def load_freetype_font_face(
    config: FontEngineRunConfig,
    lcd_compression_factor: Optional[int] = None,
):
    font_variant = config.font_variant
    if font_variant.is_ttc:
        assert font_variant.ttc_font_index is not None
        font_face = freetype.Face(str(font_variant.font_file), index=font_variant.ttc_font_index)
    else:
        font_face = freetype.Face(str(font_variant.font_file))

    font_size = estimate_font_size(config)

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


@attrs.define
class CharGlyph:
    char: str
    image: Image
    score_map: Optional[ScoreMap]
    ascent: int
    pad_up: int
    pad_down: int
    pad_left: int
    pad_right: int

    def __attrs_post_init__(self):
        # NOTE: ascent could be negative for char like '_'.
        assert self.pad_up >= 0
        assert self.pad_down >= 0
        assert self.pad_left >= 0
        assert self.pad_right >= 0

    @property
    def height(self):
        return self.image.height

    @property
    def width(self):
        return self.image.width

    def get_glyph_mask(
        self,
        box: Optional[Box] = None,
        enable_resize: bool = False,
        cv_resize_interpolation: int = cv.INTER_CUBIC,
    ):
        if self.image.mat.ndim == 2:
            # Default or monochrome.
            np_mask = (self.image.mat > 0)

        elif self.image.mat.ndim == 3:
            # LCD.
            np_mask = np.any((self.image.mat > 0), axis=2)

        else:
            raise NotImplementedError()

        mask = Mask(mat=np_mask.astype(np.uint8))
        if box:
            if mask.shape != box.shape:
                assert enable_resize
                mask = mask.to_resized_mask(
                    resized_height=box.height,
                    resized_width=box.width,
                    cv_resize_interpolation=cv_resize_interpolation,
                )
            mask = mask.to_box_attached(box)

        return mask


def trim_char_np_image_vert(np_image: np.ndarray):
    # (H, W) or (H, W, 3)
    height = np_image.shape[0]

    up = 0
    while up < height:
        all_empty = (np_image[up] == 0).all()
        if all_empty:
            up += 1
        else:
            break

    if up >= height:
        raise RuntimeError('trim_char_np_image_vert: up oob.')

    down = height - 1
    while up <= down:
        all_empty = (np_image[down] == 0).all()
        if all_empty:
            down -= 1
        else:
            break

    if down < up:
        raise RuntimeError('trim_char_np_image_vert: down oob.')

    return np_image[up:down + 1], up, height - 1 - down


def trim_char_np_image_hori(np_image: np.ndarray):
    # (H, W) or (H, W, 3)
    width = np_image.shape[1]

    left = 0
    while left < width:
        all_empty = (np_image[:, left] == 0).all()
        if all_empty:
            left += 1
        else:
            break

    if left >= width:
        raise RuntimeError('trim_char_np_image_hori: left oob.')

    right = width - 1
    while left <= right:
        all_empty = (np_image[:, right] == 0).all()
        if all_empty:
            right -= 1
        else:
            break

    if right < left:
        raise RuntimeError('trim_char_np_image_hori: right oob.')

    return np_image[:, left:right + 1], left, width - 1 - right


def build_char_glyph(
    style: FontEngineRunConfigStyle,
    char: str,
    glyph: Any,
    np_image: np.ndarray,
):
    width = np_image.shape[1]

    # "the distance from the baseline to the top-most glyph scanline"
    ascent = glyph.bitmap_top

    # "It is always zero for horizontal layouts, and positive for vertical layouts."
    assert glyph.advance.y == 0
    pad_up = 0
    pad_down = 0

    # "bitmap’s left bearing"
    pad_left = glyph.bitmap_left
    # "It is always zero for horizontal layouts, and zero for vertical ones."
    assert glyph.advance.x > 0
    pad_right = round(glyph.advance.x / 64) - pad_left - width

    # NOTE: `bitmap_left` could be negative, we simply ignore the negative value.
    pad_left = max(0, pad_left)
    pad_right = max(0, pad_right)

    assert not char.isspace()
    np_image, pad_up_inc, pad_down_inc = trim_char_np_image_vert(np_image)
    pad_up += pad_up_inc
    ascent -= pad_up_inc
    pad_down += pad_down_inc

    np_image, pad_left_inc, pad_right_inc = trim_char_np_image_hori(np_image)
    pad_left += pad_left_inc
    pad_right += pad_right_inc

    score_map = None
    if np_image.ndim == 2:
        # Gamma correction.
        np_alpha = np.power(np_image.astype(np.float32) / 255.0, style.glyph_color_gamma)
        score_map = ScoreMap(mat=np_alpha)

    return CharGlyph(
        char=char,
        image=Image(mat=np_image),
        score_map=score_map,
        ascent=ascent,
        pad_up=pad_up,
        pad_down=pad_down,
        pad_left=pad_left,
        pad_right=pad_right,
    )


def render_char_glyphs_from_text(
    config: FontEngineRunConfig,
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

        char_glyphs.append(func_render_char_glyph(config, font_face, char))

        if idx == 0 and num_spaces > 0:
            raise RuntimeError('Leading space(s) detected.')
        prev_num_spaces_for_char_glyphs.append(num_spaces)
        num_spaces = 0

    if num_spaces > 0:
        raise RuntimeError('Tailing space(s) detected.')

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
    config: FontEngineRunConfig,
    char_glyphs: Sequence[CharGlyph],
    prev_num_spaces_for_char_glyphs: Sequence[int],
    kerning_limits: Sequence[int],
    rng: RandomGenerator,
):
    style = config.style

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
        char_glyph.ascent + char_glyph.pad_up for char_glyph in char_glyphs
    )
    ascent_plus_pad_up_max = max(ascent_plus_pad_up_max, get_ref_ascent_plus_pad_up_min(config))

    text_line_height = get_ref_char_height(config)

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
    config: FontEngineRunConfig,
    char_glyphs: Sequence[CharGlyph],
    prev_num_spaces_for_char_glyphs: Sequence[int],
    rng: RandomGenerator,
):
    style = config.style

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
        char_glyph.pad_left + char_glyph.width + char_glyph.pad_right for char_glyph in char_glyphs
    )
    text_line_width = max(text_line_width, get_ref_char_width(config))

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
    config: FontEngineRunConfig,
    cv_resize_interpolation_enlarge: int,
    cv_resize_interpolation_shrink: int,
    image: Image,
    mask: Mask,
    score_map: Optional[ScoreMap],
    char_boxes: Sequence[CharBox],
    char_glyphs: Sequence[CharGlyph],
):
    # Resize if image height too small or too large.
    is_too_small = (image.height / config.height < 0.8)
    is_too_large = (image.height > config.height)

    cv_resize_interpolation = cv_resize_interpolation_enlarge
    if is_too_large:
        cv_resize_interpolation = cv_resize_interpolation_shrink

    if is_too_small or is_too_large:
        resized_image = image.to_resized_image(
            resized_height=config.height,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_mask = mask.to_resized_mask(
            resized_height=config.height,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_char_boxes = [
            char_box.to_conducted_resized_char_box(
                shapable_or_shape=image,
                resized_height=config.height,
            ) for char_box in char_boxes
        ]

        image = resized_image
        mask = resized_mask
        char_boxes = resized_char_boxes

        if score_map:
            score_map = score_map.to_resized_score_map(
                resized_height=config.height,
                cv_resize_interpolation=cv_resize_interpolation,
            )

    # Pad vertically.
    if image.height != config.height:
        pad_vert = config.height - image.height
        assert pad_vert > 0
        pad_up = pad_vert // 2
        pad_down = pad_vert - pad_up

        np_image = np.full((config.height, image.width, 3), 255, dtype=np.uint8)
        np_image[pad_up:-pad_down] = image.mat
        image.mat = np_image

        np_mask = np.zeros((config.height, image.width), dtype=np.uint8)
        np_mask[pad_up:-pad_down] = mask.mat
        mask.mat = np_mask

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
            padded_score_map = ScoreMap.from_shape((config.height, image.width))
            padded_score_map.mat[pad_up:-pad_down] = score_map.mat
            score_map = padded_score_map

    # Trim.
    if image.width > config.width:
        last_char_box_idx = len(char_boxes) - 1
        while last_char_box_idx >= 0 and char_boxes[last_char_box_idx].right >= config.width:
            last_char_box_idx -= 1

        if last_char_box_idx == len(char_boxes) - 1:
            # Corner case: char_boxes[-1].right < config.width but mage.width > config.width.
            # This is caused by glyph padding. The solution is to drop this char.
            last_char_box_idx -= 1

        if last_char_box_idx < 0 or char_boxes[last_char_box_idx].right >= config.width:
            # Cannot trim.
            return None, None, None, None

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
        image.mat = image.mat[:, :last_char_box_right + 1]
        mask.mat = mask.mat[:, :last_char_box_right + 1]

        if score_map:
            score_map.mat = score_map.mat[:, :last_char_box_right + 1]

    return image, mask, score_map, char_boxes


def resize_and_trim_text_line_vert_default(
    config: FontEngineRunConfig,
    cv_resize_interpolation_enlarge: int,
    cv_resize_interpolation_shrink: int,
    image: Image,
    mask: Mask,
    score_map: Optional[ScoreMap],
    char_boxes: Sequence[CharBox],
):
    # Resize if image width too small or too large.
    is_too_small = (image.width / config.width < 0.8)
    is_too_large = (image.width > config.width)

    cv_resize_interpolation = cv_resize_interpolation_enlarge
    if is_too_large:
        cv_resize_interpolation = cv_resize_interpolation_shrink

    if is_too_small or is_too_large:
        resized_image = image.to_resized_image(
            resized_width=config.width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_mask = mask.to_resized_mask(
            resized_width=config.width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_char_boxes = [
            char_box.to_conducted_resized_char_box(
                shapable_or_shape=image,
                resized_width=config.width,
            ) for char_box in char_boxes
        ]

        image = resized_image
        mask = resized_mask
        char_boxes = resized_char_boxes

        if score_map:
            score_map = score_map.to_resized_score_map(
                resized_width=config.width,
                cv_resize_interpolation=cv_resize_interpolation,
            )

    # Pad horizontally.
    if image.width != config.width:
        pad_hori = config.width - image.width
        assert pad_hori > 0
        pad_left = pad_hori // 2
        pad_right = pad_hori - pad_left

        np_image = np.full((image.height, config.width, 3), 255, dtype=np.uint8)
        np_image[:, pad_left:-pad_right] = image.mat
        image.mat = np_image

        np_mask = np.zeros((image.height, config.width), dtype=np.uint8)
        np_mask[:, pad_left:-pad_right] = mask.mat
        mask.mat = np_mask

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
            padded_score_map = ScoreMap.from_shape((image.height, config.width))
            padded_score_map.mat[:, pad_left:-pad_right] = score_map.mat
            score_map = padded_score_map

    # Trim.
    if image.height > config.height:
        last_char_box_idx = len(char_boxes) - 1
        while last_char_box_idx >= 0 and char_boxes[last_char_box_idx].down >= config.height:
            last_char_box_idx -= 1

        if last_char_box_idx == len(char_boxes) - 1:
            last_char_box_idx -= 1

        if last_char_box_idx < 0 or char_boxes[last_char_box_idx].down >= config.height:
            # Cannot trim.
            return None, None, None, None

        last_char_box_down = char_boxes[last_char_box_idx].down
        char_boxes = char_boxes[:last_char_box_idx + 1]
        image.mat = image.mat[:last_char_box_down + 1]
        mask.mat = mask.mat[:last_char_box_down + 1]

        if score_map:
            score_map.mat = score_map.mat[:last_char_box_down + 1]

    return image, mask, score_map, char_boxes


def render_text_line_meta(
    config: FontEngineRunConfig,
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
        config=config,
        font_face=font_face,
        func_render_char_glyph=func_render_char_glyph,
        chars=config.chars,
    )
    if not char_glyphs:
        return None

    if config.glyph_sequence == FontEngineRunConfigGlyphSequence.HORI_DEFAULT:
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
            config=config,
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
        ) = resize_and_trim_text_line_hori_default(
            config=config,
            cv_resize_interpolation_enlarge=cv_resize_interpolation_enlarge,
            cv_resize_interpolation_shrink=cv_resize_interpolation_shrink,
            image=image,
            mask=mask,
            score_map=score_map,
            char_boxes=char_boxes,
            char_glyphs=char_glyphs,
        )
        is_hori = True

    elif config.glyph_sequence == FontEngineRunConfigGlyphSequence.VERT_DEFAULT:
        # NOTE: No kerning limit detection for VERT_DEFAULT mode.
        (
            image,
            mask,
            score_map,
            char_boxes,
        ) = place_char_glyphs_in_text_line_vert_default(
            config=config,
            char_glyphs=char_glyphs,
            prev_num_spaces_for_char_glyphs=prev_num_spaces_for_char_glyphs,
            rng=rng,
        )
        (
            image,
            mask,
            score_map,
            char_boxes,
        ) = resize_and_trim_text_line_vert_default(
            config=config,
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
        while char_idx < len(config.chars) and non_space_count < len(char_boxes):
            if not config.chars[char_idx].isspace():
                non_space_count += 1
            char_idx += 1
        assert non_space_count == len(char_boxes)

        box = Box.from_shape(image.height, image.width)
        image = image.to_box_attached(box)
        mask = mask.to_box_attached(box)
        if score_map:
            score_map = score_map.to_box_attached(box)

        return TextLine(
            image=image,
            mask=mask,
            score_map=score_map,
            glyph_color=config.style.glyph_color,
            char_boxes=char_boxes,
            font_size=estimate_font_size(config),
            ref_char_height=get_ref_char_height(config),
            ref_char_width=get_ref_char_width(config),
            text=''.join(config.chars[:char_idx]),
            is_hori=is_hori,
            font_variant=config.font_variant if config.return_font_variant else None,
        )


class FreetypeDefaultFontEngine(
    Engine[
        NoneTypeEngineConfig,
        NoneTypeEngineResource,
        FontEngineRunConfig,
        Optional[TextLine],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'freetype_default'

    @staticmethod
    def render_char_glyph(
        config: FontEngineRunConfig,
        font_face: freetype.Face,
        char: str,
    ):
        load_char_flags = freetype.FT_LOAD_RENDER  # type: ignore
        if config.style.freetype_force_autohint:
            load_char_flags |= freetype.FT_LOAD_FORCE_AUTOHINT  # type: ignore
        font_face.load_char(char, load_char_flags)

        glyph = font_face.glyph
        bitmap = glyph.bitmap

        height = bitmap.rows
        width = bitmap.width
        assert width == bitmap.pitch

        # (H, W), [0, 255]
        np_image = np.array(bitmap.buffer, dtype=np.uint8).reshape(height, width)

        return build_char_glyph(config.style, char, glyph, np_image)

    def run(self, config: FontEngineRunConfig, rng: RandomGenerator) -> Optional[TextLine]:
        font_face = load_freetype_font_face(config)
        return render_text_line_meta(
            config=config,
            font_face=font_face,
            func_render_char_glyph=FreetypeDefaultFontEngine.render_char_glyph,
            rng=rng,
            cv_resize_interpolation_enlarge=sample_cv_resize_interpolation(rng),
            cv_resize_interpolation_shrink=sample_cv_resize_interpolation(
                rng,
                include_cv_inter_area=True,
            ),
        )


class FreetypeLcdFontEngine(
    Engine[
        NoneTypeEngineConfig,
        NoneTypeEngineResource,
        FontEngineRunConfig,
        Optional[TextLine],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'freetype_lcd'

    @staticmethod
    def render_char_glyph(
        config: FontEngineRunConfig,
        font_face: freetype.Face,
        lcd_hc_matrix: freetype.Matrix,
        char: str,
    ):
        load_char_flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_LCD  # type: ignore
        if config.style.freetype_force_autohint:
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
        np_image = np.array(bitmap.buffer, dtype=np.uint8).reshape(height, pitch)
        np_image = np_image[:, :width * 3].reshape(height, width, 3)

        return build_char_glyph(config.style, char, glyph, np_image)

    @staticmethod
    def bind_render_char_glyph(lcd_hc_matrix: freetype.Matrix):
        return lambda config, font_face, char: FreetypeLcdFontEngine.render_char_glyph(
            config,
            font_face,
            lcd_hc_matrix,
            char,
        )

    def run(self, config: FontEngineRunConfig, rng: RandomGenerator) -> Optional[TextLine]:
        lcd_compression_factor = 10
        font_face = load_freetype_font_face(
            config,
            lcd_compression_factor=lcd_compression_factor,
        )
        lcd_hc_matrix = build_freetype_font_face_lcd_hc_matrix(lcd_compression_factor)
        return render_text_line_meta(
            config=config,
            font_face=font_face,
            func_render_char_glyph=FreetypeLcdFontEngine.bind_render_char_glyph(lcd_hc_matrix),
            rng=rng,
            cv_resize_interpolation_enlarge=sample_cv_resize_interpolation(rng),
            cv_resize_interpolation_shrink=sample_cv_resize_interpolation(
                rng,
                include_cv_inter_area=True,
            ),
        )


class FreetypeMonochromeFontEngine(
    Engine[
        NoneTypeEngineConfig,
        NoneTypeEngineResource,
        FontEngineRunConfig,
        Optional[TextLine],
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'freetype_monochrome'

    @staticmethod
    def render_char_glyph(
        config: FontEngineRunConfig,
        font_face: freetype.Face,
        char: str,
    ):
        load_char_flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO  # type: ignore
        if config.style.freetype_force_autohint:
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

        np_image = np.array(data, dtype=np.uint8)
        assert np_image.shape == (height, width)

        return build_char_glyph(config.style, char, glyph, np_image)

    def run(self, config: FontEngineRunConfig, rng: RandomGenerator) -> Optional[TextLine]:
        font_face = load_freetype_font_face(config)
        return render_text_line_meta(
            config=config,
            font_face=font_face,
            func_render_char_glyph=FreetypeMonochromeFontEngine.render_char_glyph,
            rng=rng,
            cv_resize_interpolation_enlarge=cv.INTER_NEAREST_EXACT,
            cv_resize_interpolation_shrink=cv.INTER_NEAREST_EXACT,
        )


def debug():
    from numpy.random import default_rng
    from vkit.utility import get_data_folder
    import iolite as io
    fd = io.folder(get_data_folder(__file__), touch=True)

    from .type import FontCollection
    import os.path
    font_collection = FontCollection.from_folder(
        os.path.expandvars('$VKIT_PRIVATE_DATA/vkit_font/font_collection')
    )
    name_to_font_meta = {font_meta.name: font_meta for font_meta in font_collection.font_metas}
    # font_meta = name_to_font_meta['方正书宋简体']
    # font_meta = name_to_font_meta['STXihei']
    font_meta = name_to_font_meta['NotoSansSC']

    from vkit.engine.interface import EngineFactory

    config = FontEngineRunConfig(
        height=12,
        width=640,
        # height=640,
        # width=32,
        chars=list('我可以吞下玻璃，且不伤害到自-己??'),
        font_variant=font_meta.get_font_variant(1),
        # chars=list('this is good.'),
        # chars=[],
        # style=FontEngineRenderTextLineStyle(
        #     glyph_sequence=FontEngineRenderTextLineStyleGlyphSequence.VERT_DEFAULT
        # ),
    )
    rng = default_rng(42)
    # result = EngineFactory(FreetypeDefaultFontEngine).create().run(config, rng)
    result = EngineFactory(FreetypeMonochromeFontEngine).create().run(config, rng)
    assert result is not None
    result.image.to_file(fd / 'image.png')

    from vkit.element import Painter

    painter = Painter.create(result.mask)
    painter.paint_mask(result.mask, alpha=1.0)
    painter.to_file(fd / 'mask.png')

    painter = Painter.create(result.image)
    painter.paint_char_boxes(result.char_boxes)
    painter.to_file(fd / 'char_boxes.png')


def debug_ratio():
    from numpy.random import default_rng
    from .type import FontCollection
    import os.path
    font_collection = FontCollection.from_folder(
        os.path.expandvars('$VKIT_PRIVATE_DATA/vkit_font/font_collection')
    )
    name_to_font_meta = {font_meta.name: font_meta for font_meta in font_collection.font_metas}
    font_meta = name_to_font_meta['方正书宋简体']

    from vkit.engine.interface import EngineFactory

    font_variant = font_meta.get_font_variant(0)
    shapes = []
    engine = EngineFactory(FreetypeDefaultFontEngine).create()
    rng = default_rng(0)
    from tqdm import tqdm
    for char in tqdm(font_meta.chars):
        config = FontEngineRunConfig(
            height=32,
            width=64,
            chars=[char],
            font_variant=font_variant,
        )
        result = engine.run(config, rng)
        assert result
        assert len(result.char_boxes) == 1
        char_box = result.char_boxes[0]
        shapes.append((char_box.height, char_box.width))

    ratios = [height / width for height, width in shapes]
    print('mean', np.mean(ratios))
    print('median', np.median(ratios))
