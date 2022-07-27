from typing import List, Tuple, Optional
from enum import Enum, unique

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv

from vkit.utility import normalize_to_keys_and_probs, rng_choice
from vkit.element import Point, Mask
from vkit.engine.interface import Engine
from .type import (
    SealImpressionEngineResource,
    SealImpressionEngineRunConfig,
    CharSlot,
    SealImpressionLayout,
)


@attrs.define
class EllipseSealImpressionEngineConfig:
    # TODO: move to pipeline.
    # Shape.
    # height_ratio_min: float = 0.05
    # height_ratio_max: float = 0.15
    # weight_circle: float = 1
    # weight_general_ellipse: float = 1
    # general_ellipse_aspect_ratio_min: float = 0.75
    # general_ellipse_aspect_ratio_max: float = 1.333

    # Border.
    # TODO: add style.
    border_thickness_ratio_min: float = 0.0
    border_thickness_ratio_max: float = 0.02
    border_thickness_min: int = 1

    # Center shape.
    # TODO

    # Char slots.
    # NOTE: the ratio is relative to the height of seal impression.
    pad_ratio_min: float = 0.03
    pad_ratio_max: float = 0.08
    text_line_height_ratio_min: float = 0.025
    text_line_height_ratio_max: float = 0.175
    weight_text_line_mode_one: float = 1
    weight_text_line_mode_two: float = 1
    text_line_mode_one_gap_ratio_min: float = 0.1
    text_line_mode_one_gap_ratio_max: float = 0.7
    text_line_mode_two_gap_ratio_min: float = 0.1
    text_line_mode_two_gap_ratio_max: float = 0.4
    angle_step_ratio_min: float = 1.2
    angle_step_ratio_max: float = 2.0
    angle_step_min: int = 10

    # Color & Transparency.
    color_rgb_min: int = 128
    color_rgb_max: int = 255
    weight_color_grayscale: float = 5
    weight_color_red: float = 10
    weight_color_green: float = 1
    weight_color_blue: float = 1
    alpha_min: float = 0.25
    alpha_max: float = 0.7


@unique
class EllipseSealImpressionShapeType(Enum):
    CIRCLE = 'circle'
    GENERAL_ELLIPSE = 'general_ellipse'


@unique
class EllipseSealImpressionTextLineMode(Enum):
    ONE = 'one'
    TWO = 'two'


@unique
class EllipseSealImpressionColorMode(Enum):
    GRAYSCALE = 'grayscale'
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


class EllipseSealImpressionEngine(
    Engine[
        EllipseSealImpressionEngineConfig,
        SealImpressionEngineResource,
        SealImpressionEngineRunConfig,
        SealImpressionLayout,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'ellipse'

    def __init__(
        self,
        config: EllipseSealImpressionEngineConfig,
        resource: Optional[SealImpressionEngineResource] = None
    ):
        super().__init__(config, resource)

        # self.shape_types, self.shape_types_probs = normalize_to_keys_and_probs([
        #     (EllipseSealImpressionShapeType.CIRCLE, self.config.weight_circle),
        #     (EllipseSealImpressionShapeType.GENERAL_ELLIPSE, self.config.weight_general_ellipse),
        # ])
        self.text_line_modes, self.text_line_modes_probs = normalize_to_keys_and_probs([
            (EllipseSealImpressionTextLineMode.ONE, self.config.weight_text_line_mode_one),
            (EllipseSealImpressionTextLineMode.TWO, self.config.weight_text_line_mode_two),
        ])
        self.color_modes, self.color_modes_probs = normalize_to_keys_and_probs([
            (EllipseSealImpressionColorMode.GRAYSCALE, self.config.weight_color_grayscale),
            (EllipseSealImpressionColorMode.RED, self.config.weight_color_red),
            (EllipseSealImpressionColorMode.GREEN, self.config.weight_color_green),
            (EllipseSealImpressionColorMode.BLUE, self.config.weight_color_blue),
        ])

    # def sample_shape(self, reference_height: int, rng: RandomGenerator):
    #     # Sample height.
    #     height_ratio = float(
    #         rng.uniform(
    #             self.config.height_ratio_min,
    #             self.config.height_ratio_max,
    #         )
    #     )
    #     height = round(height_ratio * reference_height)

    #     # Make sure even.
    #     if height % 2 != 0:
    #         height += 1

    #     # Sample width.
    #     shape_type = rng_choice(rng, self.shape_types, probs=self.shape_types_probs)
    #     if shape_type == EllipseSealImpressionShapeType.CIRCLE:
    #         width = height

    #     elif shape_type == EllipseSealImpressionShapeType.GENERAL_ELLIPSE:
    #         aspect_ratio = float(
    #             rng.uniform(
    #                 self.config.general_ellipse_aspect_ratio_min,
    #                 self.config.general_ellipse_aspect_ratio_max,
    #             )
    #         )
    #         width = round(aspect_ratio * height)

    #     else:
    #         raise NotImplementedError()

    #     # Make sure even.
    #     if width % 2 != 0:
    #         width += 1

    #     return height, width

    def generate_background_mask(
        self,
        height: int,
        width: int,
        rng: RandomGenerator,
    ):
        background_mask = Mask.from_shape((height, width))

        border_thickness_ratio = float(
            rng.uniform(
                self.config.border_thickness_ratio_min,
                self.config.border_thickness_ratio_max,
            )
        )
        border_thickness = round(height * border_thickness_ratio)
        border_thickness = max(self.config.border_thickness_min, border_thickness)

        center = (width // 2, height // 2)
        axes = (width // 2 - border_thickness, height // 2 - border_thickness)
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

        return background_mask

    @staticmethod
    def sample_ellipse_points(
        ellipse_height: int,
        ellipse_width: int,
        ellipse_y_offset: int,
        ellipse_x_offset: int,
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
        points: List[Point] = []
        half_ellipse_height = ellipse_height / 2
        half_ellipse_width = ellipse_width / 2

        for x, y in unit_circle_xy_pairs:
            points.append(
                Point.create(
                    y=y * half_ellipse_height + ellipse_y_offset,
                    x=x * half_ellipse_width + ellipse_x_offset,
                )
            )
        return points

    @staticmethod
    def sample_char_slots(
        ellipse_up_height: int,
        ellipse_up_width: int,
        ellipse_down_height: int,
        ellipse_down_width: int,
        ellipse_y_offset: int,
        ellipse_x_offset: int,
        angle_begin: int,
        angle_end: int,
        angle_step: int,
        rng: RandomGenerator,
        reverse: float = False,
    ):
        char_slots: List[CharSlot] = []

        keep_last_oob = (rng.random() < 0.5)

        point_ups = EllipseSealImpressionEngine.sample_ellipse_points(
            ellipse_height=ellipse_up_height,
            ellipse_width=ellipse_up_width,
            ellipse_y_offset=ellipse_y_offset,
            ellipse_x_offset=ellipse_x_offset,
            angle_begin=angle_begin,
            angle_end=angle_end,
            angle_step=angle_step,
            keep_last_oob=keep_last_oob,
        )
        point_downs = EllipseSealImpressionEngine.sample_ellipse_points(
            ellipse_height=ellipse_down_height,
            ellipse_width=ellipse_down_width,
            ellipse_y_offset=ellipse_y_offset,
            ellipse_x_offset=ellipse_x_offset,
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

    def generate_char_slots(self, height: int, width: int, rng: RandomGenerator):
        # 1. Sample text line borders.
        pad_ratio = float(rng.uniform(
            self.config.pad_ratio_min,
            self.config.pad_ratio_max,
        ))
        text_line_height_ratio = float(
            rng.uniform(
                self.config.text_line_height_ratio_min,
                self.config.text_line_height_ratio_max,
            )
        )
        assert pad_ratio + text_line_height_ratio < 0.5

        pad = round(pad_ratio * height)
        ellipse_outer_height = height - 2 * pad
        ellipse_outer_width = width - 2 * pad
        assert ellipse_outer_height > 0 and ellipse_outer_width > 0

        text_line_height = round(text_line_height_ratio * height)
        assert text_line_height > 0
        ellipse_inner_height = ellipse_outer_height - 2 * text_line_height
        ellipse_inner_width = ellipse_outer_width - 2 * text_line_height
        assert ellipse_inner_height > 0 and ellipse_inner_width > 0

        ellipse_y_offset = height // 2
        ellipse_x_offset = width // 2

        # 2. Sample char slots.
        char_slots: List[CharSlot] = []

        radius_ref = ellipse_y_offset
        angle_step_ref = 360 * text_line_height / (2 * np.pi * radius_ref)
        angle_step_ratio = rng.uniform(
            self.config.angle_step_ratio_min,
            self.config.angle_step_ratio_max,
        )
        angle_step = max(self.config.angle_step_min, round(angle_step_ref * angle_step_ratio))

        text_line_mode = rng_choice(rng, self.text_line_modes, probs=self.text_line_modes_probs)

        if text_line_mode == EllipseSealImpressionTextLineMode.ONE:
            gap_ratio = float(
                rng.uniform(
                    self.config.text_line_mode_one_gap_ratio_min,
                    self.config.text_line_mode_one_gap_ratio_max,
                )
            )
            angle_gap = round(gap_ratio * 360)
            angle_range = 360 - angle_gap
            angle_begin = 90 + angle_gap // 2
            angle_end = angle_begin + angle_range - 1

            char_slots.extend(
                self.sample_char_slots(
                    ellipse_up_height=ellipse_outer_height,
                    ellipse_up_width=ellipse_outer_width,
                    ellipse_down_height=ellipse_inner_height,
                    ellipse_down_width=ellipse_inner_width,
                    ellipse_y_offset=ellipse_y_offset,
                    ellipse_x_offset=ellipse_x_offset,
                    angle_begin=angle_begin,
                    angle_end=angle_end,
                    angle_step=angle_step,
                    rng=rng,
                )
            )

        elif text_line_mode == EllipseSealImpressionTextLineMode.TWO:
            gap_ratio = float(
                rng.uniform(
                    self.config.text_line_mode_two_gap_ratio_min,
                    self.config.text_line_mode_two_gap_ratio_max,
                )
            )
            half_gap = round(gap_ratio * 360 / 2)

            # Up line.
            char_slots.extend(
                self.sample_char_slots(
                    ellipse_up_height=ellipse_outer_height,
                    ellipse_up_width=ellipse_outer_width,
                    ellipse_down_height=ellipse_inner_height,
                    ellipse_down_width=ellipse_inner_width,
                    ellipse_y_offset=ellipse_y_offset,
                    ellipse_x_offset=ellipse_x_offset,
                    angle_begin=180 + half_gap,
                    angle_end=360 - half_gap,
                    angle_step=angle_step,
                    rng=rng,
                )
            )

            # Down line.
            char_slots.extend(
                self.sample_char_slots(
                    ellipse_up_height=ellipse_inner_height,
                    ellipse_up_width=ellipse_inner_width,
                    ellipse_down_height=ellipse_outer_height,
                    ellipse_down_width=ellipse_outer_width,
                    ellipse_y_offset=ellipse_y_offset,
                    ellipse_x_offset=ellipse_x_offset,
                    angle_begin=half_gap,
                    angle_end=180 - half_gap,
                    angle_step=angle_step,
                    rng=rng,
                    reverse=True,
                )
            )

        else:
            raise NotImplementedError()

        return text_line_height, char_slots

    def sample_alpha_and_color(self, rng: RandomGenerator):
        alpha = float(rng.uniform(
            self.config.alpha_min,
            self.config.alpha_max,
        ))

        color_mode = rng_choice(rng, self.color_modes, probs=self.color_modes_probs)
        rgb_value = int(rng.integers(
            self.config.color_rgb_min,
            self.config.color_rgb_max + 1,
        ))
        if color_mode == EllipseSealImpressionColorMode.GRAYSCALE:
            color = (rgb_value,) * 3
        else:
            if color_mode == EllipseSealImpressionColorMode.RED:
                color = (rgb_value, 0, 0)
            elif color_mode == EllipseSealImpressionColorMode.GREEN:
                color = (0, rgb_value, 0)
            elif color_mode == EllipseSealImpressionColorMode.BLUE:
                color = (0, 0, rgb_value)
            else:
                raise NotImplementedError()

        return alpha, color

    def run(self, config: SealImpressionEngineRunConfig, rng: RandomGenerator):
        # TODO: rename all to run_config.
        alpha, color = self.sample_alpha_and_color(rng)
        background_mask = self.generate_background_mask(
            height=config.height,
            width=config.width,
            rng=rng,
        )
        text_line_height, char_slots = self.generate_char_slots(
            height=config.height,
            width=config.width,
            rng=rng,
        )

        return SealImpressionLayout(
            alpha=alpha,
            color=color,
            background_mask=background_mask,
            text_line_height=text_line_height,
            char_slots=char_slots,
        )
