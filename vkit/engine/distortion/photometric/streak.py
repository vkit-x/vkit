from typing import Tuple, Optional, List

import attrs
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.element import Image, Mask, Box
from ..interface import DistortionConfig, DistortionNopState, Distortion


@attrs.define
class LineStreakConfig(DistortionConfig):
    thickness: int = 1
    gap: int = 4
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
    mask = Mask.from_shapable(image)

    step = config.thickness + config.gap
    if config.enable_vert:
        for x_offset in range(config.thickness):
            mask.mat[:, x_offset::step] = 1
    if config.enable_hori:
        for y_offset in range(config.thickness):
            mask.mat[y_offset::step] = 1

    image = image.copy()
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
    for box in boxes:
        inner_up = box.down - config.thickness + 1
        inner_down = box.up + config.thickness - 1
        inner_left = box.right - config.thickness + 1
        inner_right = box.left + config.thickness - 1

        # Shared by top/bottom bars.
        bar_left = max(0, inner_right + 1)
        bar_right = min(image.width - 1, inner_left - 1)

        # Top bar.
        bar_up = max(0, box.up)
        bar_down = inner_down
        if 0 <= inner_down < image.height and bar_left <= bar_right:
            mask.mat[bar_up:bar_down + 1, bar_left:bar_right + 1] = 1

        # Bottom bar.
        bar_up = inner_up
        bar_down = min(image.height - 1, box.down)
        if 0 <= bar_up < image.height and bar_left <= bar_right:
            mask.mat[bar_up:bar_down + 1, bar_left:bar_right + 1] = 1

        # Shared by left/right bars.
        bar_up = max(0, box.up)
        bar_down = min(image.height - 1, box.down)

        # Left bar.
        bar_left = max(0, box.left)
        bar_right = inner_right
        if 0 <= bar_right < image.width and bar_up <= bar_down:
            mask.mat[bar_up:bar_down + 1, bar_left:bar_right + 1] = 1

        # Right bar.
        bar_left = inner_left
        bar_right = min(image.width - 1, box.right)
        if 0 <= bar_left < image.width and bar_up <= bar_down:
            mask.mat[bar_up:bar_down + 1, bar_left:bar_right + 1] = 1

    image = image.copy()
    mask.fill_image(image, config.color, alpha=config.alpha)
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
        mask.mat = cv.ellipse(
            mask.mat,
            center,
            (box.width // 2, box.height // 2),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=config.thickness,
        )

    image = image.copy()
    mask.fill_image(image, config.color, alpha=config.alpha)
    return image


ellipse_streak = Distortion(
    config_cls=EllipseStreakConfig,
    state_cls=DistortionNopState[EllipseStreakConfig],
    func_image=ellipse_streak_image,
)
