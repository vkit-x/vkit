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
from typing import List, Tuple, Optional, Sequence
from enum import Enum, unique

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv

from vkit.utility import normalize_to_keys_and_probs, rng_choice
from vkit.element import Point, PointList, Box, Mask, ImageMode
from vkit.engine.interface import (
    NoneTypeEngineInitResource,
    Engine,
    EngineExecutorFactory,
)
from vkit.engine.image import image_selector_engine_executor_factory
from .type import (
    SealImpressionEngineRunConfig,
    CharSlot,
    TextLineSlot,
    SealImpression,
)


@attrs.define
class SealImpressionEllipseEngineInitConfig:
    # Color & Transparency.
    color_rgb_min: int = 128
    color_rgb_max: int = 255
    weight_color_grayscale: float = 5
    weight_color_red: float = 10
    weight_color_green: float = 1
    weight_color_blue: float = 1
    alpha_min: float = 0.25
    alpha_max: float = 0.75

    # Border.
    border_thickness_ratio_min: float = 0.0
    border_thickness_ratio_max: float = 0.03
    border_thickness_min: int = 2
    weight_border_style_solid_line = 3
    weight_border_style_double_lines = 1

    # Char slots.
    # NOTE: the ratio is relative to the height of seal impression.
    pad_ratio_min: float = 0.03
    pad_ratio_max: float = 0.08
    text_line_height_ratio_min: float = 0.075
    text_line_height_ratio_max: float = 0.2
    weight_text_line_mode_one: float = 1
    weight_text_line_mode_two: float = 1
    text_line_mode_one_gap_ratio_min: float = 0.1
    text_line_mode_one_gap_ratio_max: float = 0.55
    text_line_mode_two_gap_ratio_min: float = 0.1
    text_line_mode_two_gap_ratio_max: float = 0.4
    char_aspect_ratio_min: float = 0.4
    char_aspect_ratio_max: float = 0.9
    char_space_ratio_min: float = 0.05
    char_space_ratio_max: float = 0.25
    angle_step_min: int = 10

    # Icon.
    icon_image_folders: Optional[Sequence[str]] = None
    icon_image_grayscale_min: int = 127
    prob_add_icon: float = 0.9
    icon_height_ratio_min: float = 0.35
    icon_height_ratio_max: float = 0.75
    icon_width_ratio_min: float = 0.35
    icon_width_ratio_max: float = 0.75

    # Internal text line.
    prob_add_internal_text_line: float = 0.5
    internal_text_line_height_ratio_min: float = 0.075
    internal_text_line_height_ratio_max: float = 0.15
    internal_text_line_width_ratio_min: float = 0.22
    internal_text_line_width_ratio_max: float = 0.5


@unique
class SealImpressionEllipseBorderStyle(Enum):
    SOLID_LINE = 'solid_line'
    DOUBLE_LINES = 'double_lines'


@unique
class SealImpressionEllipseTextLineMode(Enum):
    ONE = 'one'
    TWO = 'two'


@unique
class SealImpressionEllipseColorMode(Enum):
    GRAYSCALE = 'grayscale'
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


@attrs.define
class TextLineRoughPlacement:
    ellipse_outer_height: int
    ellipse_outer_width: int
    ellipse_inner_height: int
    ellipse_inner_width: int
    text_line_height: int
    angle_begin: int
    angle_end: int
    clockwise: bool


class SealImpressionEllipseEngine(
    Engine[
        SealImpressionEllipseEngineInitConfig,
        NoneTypeEngineInitResource,
        SealImpressionEngineRunConfig,
        SealImpression,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'ellipse'

    def __init__(
        self,
        init_config: SealImpressionEllipseEngineInitConfig,
        init_resource: Optional[NoneTypeEngineInitResource] = None
    ):
        super().__init__(init_config, init_resource)

        self.border_styles, self.border_styles_probs = normalize_to_keys_and_probs([
            (
                SealImpressionEllipseBorderStyle.SOLID_LINE,
                self.init_config.weight_border_style_solid_line,
            ),
            (
                SealImpressionEllipseBorderStyle.DOUBLE_LINES,
                self.init_config.weight_border_style_double_lines,
            ),
        ])
        self.text_line_modes, self.text_line_modes_probs = normalize_to_keys_and_probs([
            (
                SealImpressionEllipseTextLineMode.ONE,
                self.init_config.weight_text_line_mode_one,
            ),
            (
                SealImpressionEllipseTextLineMode.TWO,
                self.init_config.weight_text_line_mode_two,
            ),
        ])
        self.color_modes, self.color_modes_probs = normalize_to_keys_and_probs([
            (
                SealImpressionEllipseColorMode.GRAYSCALE,
                self.init_config.weight_color_grayscale,
            ),
            (
                SealImpressionEllipseColorMode.RED,
                self.init_config.weight_color_red,
            ),
            (
                SealImpressionEllipseColorMode.GREEN,
                self.init_config.weight_color_green,
            ),
            (
                SealImpressionEllipseColorMode.BLUE,
                self.init_config.weight_color_blue,
            ),
        ])
        self.icon_image_selector = None
        if self.init_config.icon_image_folders:
            self.icon_image_selector = image_selector_engine_executor_factory.create({
                'image_folders': self.init_config.icon_image_folders,
                'target_image_mode': ImageMode.GRAYSCALE,
                'force_resize': True,
            })

    def sample_alpha_and_color(self, rng: RandomGenerator):
        alpha = float(rng.uniform(
            self.init_config.alpha_min,
            self.init_config.alpha_max,
        ))

        color_mode = rng_choice(rng, self.color_modes, probs=self.color_modes_probs)
        rgb_value = int(
            rng.integers(
                self.init_config.color_rgb_min,
                self.init_config.color_rgb_max + 1,
            )
        )
        if color_mode == SealImpressionEllipseColorMode.GRAYSCALE:
            color = (rgb_value,) * 3
        else:
            if color_mode == SealImpressionEllipseColorMode.RED:
                color = (rgb_value, 0, 0)
            elif color_mode == SealImpressionEllipseColorMode.GREEN:
                color = (0, rgb_value, 0)
            elif color_mode == SealImpressionEllipseColorMode.BLUE:
                color = (0, 0, rgb_value)
            else:
                raise NotImplementedError()

        return alpha, color

    @classmethod
    def sample_ellipse_points(
        cls,
        ellipse_height: int,
        ellipse_width: int,
        ellipse_offset_y: int,
        ellipse_offset_x: int,
        angle_begin: int,
        angle_end: int,
        angle_step: int,
        keep_last_oob: bool,
    ):
        # Sample points in unit circle.
        unit_circle_xy_pairs: List[Tuple[float, float]] = []
        angle = angle_begin
        while angle <= angle_end or (keep_last_oob and angle - angle_end < angle_step):
            theta = angle / 180 * np.pi
            # Shifted.
            x = float(np.cos(theta))
            y = float(np.sin(theta))
            unit_circle_xy_pairs.append((x, y))
            # Move forward.
            angle += angle_step

        # Reshape to ellipse.
        points = PointList()
        half_ellipse_height = ellipse_height / 2
        half_ellipse_width = ellipse_width / 2
        for x, y in unit_circle_xy_pairs:
            points.append(
                Point.create(
                    y=y * half_ellipse_height + ellipse_offset_y,
                    x=x * half_ellipse_width + ellipse_offset_x,
                )
            )
        return points

    @classmethod
    def sample_char_slots(
        cls,
        ellipse_up_height: int,
        ellipse_up_width: int,
        ellipse_down_height: int,
        ellipse_down_width: int,
        ellipse_offset_y: int,
        ellipse_offset_x: int,
        angle_begin: int,
        angle_end: int,
        angle_step: int,
        rng: RandomGenerator,
        reverse: float = False,
    ):
        char_slots: List[CharSlot] = []

        keep_last_oob = (rng.random() < 0.5)

        point_ups = cls.sample_ellipse_points(
            ellipse_height=ellipse_up_height,
            ellipse_width=ellipse_up_width,
            ellipse_offset_y=ellipse_offset_y,
            ellipse_offset_x=ellipse_offset_x,
            angle_begin=angle_begin,
            angle_end=angle_end,
            angle_step=angle_step,
            keep_last_oob=keep_last_oob,
        )
        point_downs = cls.sample_ellipse_points(
            ellipse_height=ellipse_down_height,
            ellipse_width=ellipse_down_width,
            ellipse_offset_y=ellipse_offset_y,
            ellipse_offset_x=ellipse_offset_x,
            angle_begin=angle_begin,
            angle_end=angle_end,
            angle_step=angle_step,
            keep_last_oob=keep_last_oob,
        )
        for point_up, point_down in zip(point_ups, point_downs):
            char_slots.append(CharSlot.build(point_up=point_up, point_down=point_down))

        if reverse:
            char_slots = list(reversed(char_slots))

        return char_slots

    def sample_curved_text_line_rough_placements(
        self,
        height: int,
        width: int,
        rng: RandomGenerator,
    ):
        # Shared outer ellipse.
        pad_ratio = float(
            rng.uniform(
                self.init_config.pad_ratio_min,
                self.init_config.pad_ratio_max,
            )
        )

        pad = round(pad_ratio * height)
        ellipse_outer_height = height - 2 * pad
        ellipse_outer_width = width - 2 * pad
        assert ellipse_outer_height > 0 and ellipse_outer_width > 0

        # Rough placements.
        rough_placements: List[TextLineRoughPlacement] = []

        # Place text line one.
        half_gap = None
        text_line_mode = rng_choice(rng, self.text_line_modes, probs=self.text_line_modes_probs)

        if text_line_mode == SealImpressionEllipseTextLineMode.ONE:
            gap_ratio = float(
                rng.uniform(
                    self.init_config.text_line_mode_one_gap_ratio_min,
                    self.init_config.text_line_mode_one_gap_ratio_max,
                )
            )
            angle_gap = round(gap_ratio * 360)
            angle_range = 360 - angle_gap
            text_line_one_angle_begin = 90 + angle_gap // 2
            text_line_one_angle_end = text_line_one_angle_begin + angle_range - 1

        elif text_line_mode == SealImpressionEllipseTextLineMode.TWO:
            gap_ratio = float(
                rng.uniform(
                    self.init_config.text_line_mode_two_gap_ratio_min,
                    self.init_config.text_line_mode_two_gap_ratio_max,
                )
            )
            half_gap = round(gap_ratio * 360 / 2)

            text_line_one_angle_begin = 180 + half_gap
            text_line_one_angle_end = 360 - half_gap

        else:
            raise NotImplementedError()

        text_line_one_height_ratio = float(
            rng.uniform(
                self.init_config.text_line_height_ratio_min,
                self.init_config.text_line_height_ratio_max,
            )
        )
        text_line_one_height = round(text_line_one_height_ratio * height)
        assert text_line_one_height > 0
        ellipse_inner_one_height = ellipse_outer_height - 2 * text_line_one_height
        ellipse_inner_one_width = ellipse_outer_width - 2 * text_line_one_height
        assert ellipse_inner_one_height > 0 and ellipse_inner_one_width > 0

        rough_placements.append(
            TextLineRoughPlacement(
                ellipse_outer_height=ellipse_outer_height,
                ellipse_outer_width=ellipse_outer_width,
                ellipse_inner_height=ellipse_inner_one_height,
                ellipse_inner_width=ellipse_inner_one_width,
                text_line_height=text_line_one_height,
                angle_begin=text_line_one_angle_begin,
                angle_end=text_line_one_angle_end,
                clockwise=True,
            )
        )

        # Now for the text line two.
        if text_line_mode == SealImpressionEllipseTextLineMode.TWO:
            assert half_gap

            text_line_two_height_ratio = float(
                rng.uniform(
                    self.init_config.text_line_height_ratio_min,
                    self.init_config.text_line_height_ratio_max,
                )
            )
            text_line_two_height = round(text_line_two_height_ratio * height)
            assert text_line_two_height > 0
            ellipse_inner_two_height = ellipse_outer_height - 2 * text_line_two_height
            ellipse_inner_two_width = ellipse_outer_width - 2 * text_line_two_height
            assert ellipse_inner_two_height > 0 and ellipse_inner_two_width > 0

            text_line_two_angle_begin = half_gap
            text_line_two_angle_end = 180 - half_gap

            rough_placements.append(
                TextLineRoughPlacement(
                    ellipse_outer_height=ellipse_outer_height,
                    ellipse_outer_width=ellipse_outer_width,
                    ellipse_inner_height=ellipse_inner_two_height,
                    ellipse_inner_width=ellipse_inner_two_width,
                    text_line_height=text_line_two_height,
                    angle_begin=text_line_two_angle_begin,
                    angle_end=text_line_two_angle_end,
                    clockwise=False,
                )
            )

        return rough_placements

    def generate_text_line_slots_based_on_rough_placements(
        self,
        height: int,
        width: int,
        rough_placements: Sequence[TextLineRoughPlacement],
        rng: RandomGenerator,
    ):
        ellipse_offset_y = height // 2
        ellipse_offset_x = width // 2

        text_line_slots: List[TextLineSlot] = []

        for rough_placement in rough_placements:
            char_aspect_ratio = float(
                rng.uniform(
                    self.init_config.char_aspect_ratio_min,
                    self.init_config.char_aspect_ratio_max,
                )
            )
            char_width_ref = max(1, round(rough_placement.text_line_height * char_aspect_ratio))

            char_space_ratio = float(
                rng.uniform(
                    self.init_config.char_space_ratio_min,
                    self.init_config.char_space_ratio_max,
                )
            )
            char_space_ref = max(1, round(rough_placement.text_line_height * char_space_ratio))

            radius_ref = max(1, ellipse_offset_y)
            angle_step = max(
                self.init_config.angle_step_min,
                round(360 * (char_width_ref + char_space_ref) / (2 * np.pi * radius_ref)),
            )

            if rough_placement.clockwise:
                char_slots = self.sample_char_slots(
                    ellipse_up_height=rough_placement.ellipse_outer_height,
                    ellipse_up_width=rough_placement.ellipse_outer_width,
                    ellipse_down_height=rough_placement.ellipse_inner_height,
                    ellipse_down_width=rough_placement.ellipse_inner_width,
                    ellipse_offset_y=ellipse_offset_y,
                    ellipse_offset_x=ellipse_offset_x,
                    angle_begin=rough_placement.angle_begin,
                    angle_end=rough_placement.angle_end,
                    angle_step=angle_step,
                    rng=rng,
                )

            else:
                char_slots = self.sample_char_slots(
                    ellipse_up_height=rough_placement.ellipse_inner_height,
                    ellipse_up_width=rough_placement.ellipse_inner_width,
                    ellipse_down_height=rough_placement.ellipse_outer_height,
                    ellipse_down_width=rough_placement.ellipse_outer_width,
                    ellipse_offset_y=ellipse_offset_y,
                    ellipse_offset_x=ellipse_offset_x,
                    angle_begin=rough_placement.angle_begin,
                    angle_end=rough_placement.angle_end,
                    angle_step=angle_step,
                    rng=rng,
                    reverse=True,
                )

            text_line_slots.append(
                TextLineSlot(
                    text_line_height=rough_placement.text_line_height,
                    char_aspect_ratio=char_aspect_ratio,
                    char_slots=char_slots,
                )
            )

        return text_line_slots

    def generate_text_line_slots(self, height: int, width: int, rng: RandomGenerator):
        rough_placements = self.sample_curved_text_line_rough_placements(
            height=height,
            width=width,
            rng=rng,
        )
        text_line_slots = self.generate_text_line_slots_based_on_rough_placements(
            height=height,
            width=width,
            rough_placements=rough_placements,
            rng=rng,
        )
        ellipse_inner_shape = (
            min(rough_placement.ellipse_inner_height for rough_placement in rough_placements),
            min(rough_placement.ellipse_inner_width for rough_placement in rough_placements),
        )
        return text_line_slots, ellipse_inner_shape

    def sample_icon_box(
        self,
        height: int,
        width: int,
        ellipse_inner_shape: Tuple[int, int],
        rng: RandomGenerator,
    ):
        ellipse_inner_height, ellipse_inner_width = ellipse_inner_shape

        box_height_ratio = rng.uniform(
            self.init_config.icon_height_ratio_min,
            self.init_config.icon_height_ratio_max,
        )
        box_height = round(ellipse_inner_height * box_height_ratio)

        box_width_ratio = rng.uniform(
            self.init_config.icon_width_ratio_min,
            self.init_config.icon_width_ratio_max,
        )
        box_width = round(ellipse_inner_width * box_width_ratio)

        up = (height - box_height) // 2
        down = up + box_height - 1
        left = (width - box_width) // 2
        right = left + box_width - 1
        return Box(up=up, down=down, left=left, right=right)

    def sample_internal_text_line_box(
        self,
        height: int,
        width: int,
        ellipse_inner_shape: Tuple[int, int],
        icon_box_down: Optional[int],
        rng: RandomGenerator,
    ):
        ellipse_inner_height, ellipse_inner_width = ellipse_inner_shape
        if ellipse_inner_height > ellipse_inner_width:
            # Not supported yet.
            return None

        # Vert.
        box_height_ratio = rng.uniform(
            self.init_config.internal_text_line_height_ratio_min,
            self.init_config.internal_text_line_height_ratio_max,
        )
        box_height = round(ellipse_inner_height * box_height_ratio)

        half_height = height // 2
        up = half_height
        if icon_box_down:
            up = icon_box_down + 1
        down = min(
            height - 1,
            half_height + ellipse_inner_height // 2 - 1,
            up + box_height - 1,
        )

        if up > down:
            return None

        # Hori.
        ellipse_h = down + 1 - half_height
        ellipse_a = ellipse_inner_width / 2
        ellipse_b = ellipse_inner_height / 2
        box_width_max = round(2 * ellipse_b * np.sqrt(ellipse_a**2 - ellipse_h**2) / ellipse_a)

        box_width_ratio = rng.uniform(
            self.init_config.internal_text_line_width_ratio_min,
            self.init_config.internal_text_line_width_ratio_max,
        )
        box_width = round(ellipse_inner_width * box_width_ratio)
        box_width = max(box_width_max, box_width)

        left = (width - box_width) // 2
        right = left + box_width - 1

        if left > right:
            return None

        return Box(up=up, down=down, left=left, right=right)

    def generate_background(
        self,
        height: int,
        width: int,
        ellipse_inner_shape: Tuple[int, int],
        rng: RandomGenerator,
    ):
        background_mask = Mask.from_shape((height, width))

        border_style = rng_choice(rng, self.border_styles, probs=self.border_styles_probs)

        # Will generate solid line first.
        border_thickness_ratio = float(
            rng.uniform(
                self.init_config.border_thickness_ratio_min,
                self.init_config.border_thickness_ratio_max,
            )
        )
        border_thickness = round(height * border_thickness_ratio)
        border_thickness = max(self.init_config.border_thickness_min, border_thickness)

        center = (width // 2, height // 2)
        # NOTE: minus 1 to make sure the border is inbound.
        axes = (width // 2 - border_thickness - 1, height // 2 - border_thickness - 1)
        cv.ellipse(
            background_mask.mat,
            center=center,
            axes=axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=border_thickness,
        )

        if border_thickness > 2 * self.init_config.border_thickness_min + 1 \
                and border_style == SealImpressionEllipseBorderStyle.DOUBLE_LINES:
            # Remove the middle part to generate double lines.
            border_thickness_empty = int(
                rng.integers(
                    1,
                    border_thickness - 2 * self.init_config.border_thickness_min,
                )
            )
            cv.ellipse(
                background_mask.mat,
                center=center,
                # NOTE: I don't know why, but this works as expected.
                # Probably `axes` points to the center of border.
                axes=axes,
                angle=0,
                startAngle=0,
                endAngle=360,
                color=0,
                thickness=border_thickness_empty,
            )

        icon_box_down = None
        if self.icon_image_selector and rng.random() < self.init_config.prob_add_icon:
            icon_box = self.sample_icon_box(
                height=height,
                width=width,
                ellipse_inner_shape=ellipse_inner_shape,
                rng=rng,
            )
            icon_box_down = icon_box.down
            icon_grayscale_image = self.icon_image_selector.run(
                {
                    'height': icon_box.height,
                    'width': icon_box.width
                },
                rng,
            )
            icon_mask_mat = (icon_grayscale_image.mat > self.init_config.icon_image_grayscale_min)
            icon_mask = Mask(mat=icon_mask_mat.astype(np.uint8))
            icon_box.fill_mask(background_mask, icon_mask)

        internal_text_line_box = None
        if rng.random() < self.init_config.prob_add_internal_text_line:
            internal_text_line_box = self.sample_internal_text_line_box(
                height=height,
                width=width,
                ellipse_inner_shape=ellipse_inner_shape,
                icon_box_down=icon_box_down,
                rng=rng,
            )

        return background_mask, internal_text_line_box

    def run(self, run_config: SealImpressionEngineRunConfig, rng: RandomGenerator):
        alpha, color = self.sample_alpha_and_color(rng)
        text_line_slots, ellipse_inner_shape = self.generate_text_line_slots(
            height=run_config.height,
            width=run_config.width,
            rng=rng,
        )
        background_mask, internal_text_line_box = self.generate_background(
            height=run_config.height,
            width=run_config.width,
            ellipse_inner_shape=ellipse_inner_shape,
            rng=rng,
        )
        return SealImpression(
            alpha=alpha,
            color=color,
            background_mask=background_mask,
            text_line_slots=text_line_slots,
            internal_text_line_box=internal_text_line_box,
        )


seal_impression_ellipse_engine_executor_factory = EngineExecutorFactory(SealImpressionEllipseEngine)
