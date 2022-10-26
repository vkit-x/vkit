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
from typing import Tuple, Optional, List

import attrs
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.element import Image, Mask, Box
from ..interface import DistortionConfig, DistortionNopState, Distortion


def fill_vert_dash_gap(dash_thickness: int, dash_gap: int, mask: Mask):
    if dash_thickness <= 0 or dash_gap <= 0:
        return

    with mask.writable_context:
        step = dash_thickness + dash_gap
        for offset_y in range(dash_gap):
            mask.mat[offset_y::step] = 0


def fill_hori_dash_gap(dash_thickness: int, dash_gap: int, mask: Mask):
    if dash_thickness <= 0 or dash_gap <= 0:
        return

    with mask.writable_context:
        step = dash_thickness + dash_gap
        for offset_x in range(dash_gap):
            mask.mat[:, offset_x::step] = 0


@attrs.define
class LineStreakConfig(DistortionConfig):
    thickness: int = 1
    gap: int = 4
    dash_thickness: int = 0
    dash_gap: int = 0
    color: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 1.0
    enable_vert: bool = True
    enable_hori: bool = True


def line_streak_image(
    config: LineStreakConfig,
    state: Optional[DistortionNopState[LineStreakConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    masks: List[Mask] = []

    step = config.thickness + config.gap

    if config.enable_vert:
        mask = Mask.from_shapable(image)

        with mask.writable_context:
            for offset_x in range(config.thickness):
                mask.mat[:, offset_x::step] = 1

        fill_vert_dash_gap(
            dash_thickness=config.dash_thickness,
            dash_gap=config.dash_gap,
            mask=mask,
        )

        masks.append(mask)

    if config.enable_hori:
        mask = Mask.from_shapable(image)

        with mask.writable_context:
            for offset_y in range(config.thickness):
                mask.mat[offset_y::step] = 1

        fill_hori_dash_gap(
            dash_thickness=config.dash_thickness,
            dash_gap=config.dash_gap,
            mask=mask,
        )

        masks.append(mask)

    image = image.copy()
    for mask in masks:
        mask.fill_image(image, config.color, alpha=config.alpha)
    return image


line_streak = Distortion(
    config_cls=LineStreakConfig,
    state_cls=DistortionNopState[LineStreakConfig],
    func_image=line_streak_image,
)


def generate_centered_boxes(
    height: int,
    width: int,
    aspect_ratio: float,
    short_side_min: int,
    short_side_step: int,
):
    center_y = height // 2
    center_x = width // 2

    boxes: List[Box] = []
    idx = 0
    while True:
        short_side = short_side_min + idx * short_side_step
        if aspect_ratio >= 1:
            # hori side is longer.
            height_min = short_side
            width_min = round(height_min * aspect_ratio)
        elif 0 < aspect_ratio < 1:
            # vert side is longer.
            width_min = short_side
            height_min = round(width_min / aspect_ratio)
        else:
            raise NotImplementedError()

        up = center_y - height_min // 2
        down = up + height_min - 1
        left = center_x - width_min // 2
        right = left + width_min - 1

        if (0 <= up and down < height) or (0 <= left and right < width):
            boxes.append(Box(up=up, down=down, left=left, right=right))
            idx += 1
        else:
            break

    return boxes


@attrs.define
class RectangleStreakConfig(DistortionConfig):
    thickness: int = 1
    aspect_ratio: Optional[float] = None
    dash_thickness: int = 0
    dash_gap: int = 0
    short_side_min: int = 10
    short_side_step: int = 10
    color: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 1.0


def rectangle_streak_image(
    config: RectangleStreakConfig,
    state: Optional[DistortionNopState[RectangleStreakConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    aspect_ratio = config.aspect_ratio
    if aspect_ratio is None:
        aspect_ratio = image.width / image.height

    boxes = generate_centered_boxes(
        height=image.height,
        width=image.width,
        aspect_ratio=aspect_ratio,
        short_side_min=config.short_side_min,
        short_side_step=config.short_side_step,
    )

    # Generate bars.
    vert_bars: List[Box] = []
    hori_bars: List[Box] = []

    for box in boxes:
        inner_up = box.down - config.thickness + 1
        inner_down = box.up + config.thickness - 1
        inner_left = box.right - config.thickness + 1
        inner_right = box.left + config.thickness - 1

        # Shared by left/right bars.
        bar_up = max(0, box.up)
        bar_down = min(image.height - 1, box.down)

        # Left bar.
        bar_left = max(0, box.left)
        bar_right = inner_right

        if 0 <= bar_right < image.width and bar_up <= bar_down:
            vert_bars.append(Box(
                up=bar_up,
                down=bar_down,
                left=bar_left,
                right=bar_right,
            ))

        # Right bar.
        bar_left = inner_left
        bar_right = min(image.width - 1, box.right)

        if 0 <= bar_left < image.width and bar_up <= bar_down:
            vert_bars.append(Box(
                up=bar_up,
                down=bar_down,
                left=bar_left,
                right=bar_right,
            ))

        # Shared by top/bottom bars.
        bar_left = max(0, inner_right + 1)
        bar_right = min(image.width - 1, inner_left - 1)

        # Top bar.
        bar_up = max(0, box.up)
        bar_down = inner_down

        if 0 <= inner_down < image.height and bar_left <= bar_right:
            hori_bars.append(Box(
                up=bar_up,
                down=bar_down,
                left=bar_left,
                right=bar_right,
            ))

        # Bottom bar.
        bar_up = inner_up
        bar_down = min(image.height - 1, box.down)

        if 0 <= bar_up < image.height and bar_left <= bar_right:
            hori_bars.append(Box(
                up=bar_up,
                down=bar_down,
                left=bar_left,
                right=bar_right,
            ))

    # Render.
    mask_vert = Mask.from_shapable(image)

    with mask_vert.writable_context:
        for bar in vert_bars:
            mask_vert.mat[bar.up:bar.down + 1, bar.left:bar.right + 1] = 1

    fill_vert_dash_gap(
        dash_thickness=config.dash_thickness,
        dash_gap=config.dash_gap,
        mask=mask_vert,
    )

    mask_hori = Mask.from_shapable(image)

    with mask_hori.writable_context:
        for bar in hori_bars:
            mask_hori.mat[bar.up:bar.down + 1, bar.left:bar.right + 1] = 1

    fill_hori_dash_gap(
        dash_thickness=config.dash_thickness,
        dash_gap=config.dash_gap,
        mask=mask_hori,
    )

    image = image.copy()
    mask_vert.fill_image(image, config.color, alpha=config.alpha)
    mask_hori.fill_image(image, config.color, alpha=config.alpha)
    return image


rectangle_streak = Distortion(
    config_cls=RectangleStreakConfig,
    state_cls=DistortionNopState[RectangleStreakConfig],
    func_image=rectangle_streak_image,
)


@attrs.define
class EllipseStreakConfig(DistortionConfig):
    thickness: int = 1
    aspect_ratio: Optional[float] = None
    short_side_min: int = 10
    short_side_step: int = 10
    color: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 1.0


def ellipse_streak_image(
    config: EllipseStreakConfig,
    state: Optional[DistortionNopState[EllipseStreakConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    mask = Mask.from_shapable(image)

    aspect_ratio = config.aspect_ratio
    if aspect_ratio is None:
        aspect_ratio = image.width / image.height

    boxes = generate_centered_boxes(
        height=image.height,
        width=image.width,
        aspect_ratio=aspect_ratio,
        short_side_min=config.short_side_min,
        short_side_step=config.short_side_step,
    )
    center_y = image.height // 2
    center_x = image.width // 2
    center = (center_x, center_y)
    for box in boxes:
        mask.assign_mat(
            cv.ellipse(
                mask.mat,
                center=center,
                axes=(box.width // 2, box.height // 2),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=config.thickness,
            )
        )

    image = image.copy()
    mask.fill_image(image, config.color, alpha=config.alpha)
    return image


ellipse_streak = Distortion(
    config_cls=EllipseStreakConfig,
    state_cls=DistortionNopState[EllipseStreakConfig],
    func_image=ellipse_streak_image,
)
