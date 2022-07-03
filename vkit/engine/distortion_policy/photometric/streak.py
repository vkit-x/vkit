from typing import Tuple

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.engine import distortion
from ..type import DistortionConfigGenerator, DistortionPolicyFactory
from ..opt import sample_float


@attrs.define
class LineStreakConfigGeneratorConfig:
    thickness_min: int = 1
    thickness_max: int = 4
    gap_min: int = 5
    gap_ratio_min: float = 0.01
    gap_ratio_max: float = 0.5
    alpha_min: float = 0.2
    alpha_max: float = 1.0


class LineStreakConfigGenerator(
    DistortionConfigGenerator[
        LineStreakConfigGeneratorConfig,
        distortion.LineStreakConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        long_side_length = max(shape)
        gap_ratio = sample_float(
            level=self.level,
            value_min=self.config.gap_ratio_min,
            value_max=self.config.gap_ratio_max,
            prob_reciprocal=None,
            rng=rng,
            inverse_level=True,
        )
        gap = max(self.config.gap_min, round(gap_ratio * long_side_length))

        thickness = rng.integers(self.config.thickness_min, self.config.thickness_max + 1)
        alpha = rng.uniform(self.config.alpha_min, self.config.alpha_max)

        mode = rng.integers(0, 3)
        if mode == 0:
            enable_vert = True
            enable_hori = False
        elif mode == 1:
            enable_vert = False
            enable_hori = True
        elif mode == 2:
            enable_vert = True
            enable_hori = True
        else:
            raise NotImplementedError()

        return distortion.LineStreakConfig(
            thickness=thickness,
            gap=gap,
            alpha=alpha,
            enable_vert=enable_vert,
            enable_hori=enable_hori,
        )


line_streak_policy_factory = DistortionPolicyFactory(
    distortion.line_streak,
    LineStreakConfigGenerator,
)


def sample_params_for_rectangle_and_ellipse_streak(
    level: int,
    thickness_min: int,
    thickness_max: int,
    aspect_ratio_min: float,
    aspect_ratio_max: float,
    short_side_min: int,
    short_side_min_ratio_min: float,
    short_side_min_ratio_max: float,
    short_side_step_ratio_min: float,
    short_side_step_ratio_max: float,
    alpha_min: float,
    alpha_max: float,
    shape: Tuple[int, int],
    rng: RandomGenerator,
):
    long_side_length = max(shape)
    short_side_min_ratio = sample_float(
        level=level,
        value_min=short_side_min_ratio_min,
        value_max=short_side_min_ratio_max,
        prob_reciprocal=None,
        rng=rng,
        inverse_level=True,
    )
    short_side_min = max(
        short_side_min,
        round(short_side_min_ratio * long_side_length),
    )

    short_side_step_ratio = rng.uniform(
        short_side_step_ratio_min,
        short_side_step_ratio_max,
    )
    short_side_step = round(short_side_step_ratio * short_side_min)

    thickness = rng.integers(thickness_min, thickness_max + 1)
    aspect_ratio = rng.uniform(aspect_ratio_min, aspect_ratio_max)
    alpha = rng.uniform(alpha_min, alpha_max)

    return (
        thickness,
        aspect_ratio,
        short_side_min,
        short_side_step,
        alpha,
    )


@attrs.define
class RectangleStreakConfigGeneratorConfig:
    thickness_min: int = 1
    thickness_max: int = 4
    aspect_ratio_min: float = 0.5
    aspect_ratio_max: float = 1.5
    short_side_min: int = 5
    short_side_min_ratio_min: float = 0.01
    short_side_min_ratio_max: float = 0.25
    short_side_step_ratio_min: float = 0.8
    short_side_step_ratio_max: float = 3.0
    alpha_min: float = 0.2
    alpha_max: float = 1.0


class RectangleStreakConfigGenerator(
    DistortionConfigGenerator[
        RectangleStreakConfigGeneratorConfig,
        distortion.RectangleStreakConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        (
            thickness,
            aspect_ratio,
            short_side_min,
            short_side_step,
            alpha,
        ) = sample_params_for_rectangle_and_ellipse_streak(
            level=self.level,
            thickness_min=self.config.thickness_min,
            thickness_max=self.config.thickness_max,
            aspect_ratio_min=self.config.aspect_ratio_min,
            aspect_ratio_max=self.config.aspect_ratio_max,
            short_side_min=self.config.short_side_min,
            short_side_min_ratio_min=self.config.short_side_min_ratio_min,
            short_side_min_ratio_max=self.config.short_side_min_ratio_max,
            short_side_step_ratio_min=self.config.short_side_step_ratio_min,
            short_side_step_ratio_max=self.config.short_side_step_ratio_max,
            alpha_min=self.config.alpha_min,
            alpha_max=self.config.alpha_max,
            shape=shape,
            rng=rng,
        )

        return distortion.RectangleStreakConfig(
            thickness=thickness,
            aspect_ratio=aspect_ratio,
            short_side_min=short_side_min,
            short_side_step=short_side_step,
            alpha=alpha,
        )


rectangle_streak_policy_factory = DistortionPolicyFactory(
    distortion.rectangle_streak,
    RectangleStreakConfigGenerator,
)


@attrs.define
class EllipseStreakConfigGeneratorConfig:
    thickness_min: int = 1
    thickness_max: int = 3
    aspect_ratio_min: float = 0.5
    aspect_ratio_max: float = 1.5
    short_side_min: int = 5
    short_side_min_ratio_min: float = 0.01
    short_side_min_ratio_max: float = 0.25
    short_side_step_ratio_min: float = 0.8
    short_side_step_ratio_max: float = 3.0
    alpha_min: float = 0.2
    alpha_max: float = 1.0


class EllipseStreakConfigGenerator(
    DistortionConfigGenerator[
        EllipseStreakConfigGeneratorConfig,
        distortion.EllipseStreakConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        (
            thickness,
            aspect_ratio,
            short_side_min,
            short_side_step,
            alpha,
        ) = sample_params_for_rectangle_and_ellipse_streak(
            level=self.level,
            thickness_min=self.config.thickness_min,
            thickness_max=self.config.thickness_max,
            aspect_ratio_min=self.config.aspect_ratio_min,
            aspect_ratio_max=self.config.aspect_ratio_max,
            short_side_min=self.config.short_side_min,
            short_side_min_ratio_min=self.config.short_side_min_ratio_min,
            short_side_min_ratio_max=self.config.short_side_min_ratio_max,
            short_side_step_ratio_min=self.config.short_side_step_ratio_min,
            short_side_step_ratio_max=self.config.short_side_step_ratio_max,
            alpha_min=self.config.alpha_min,
            alpha_max=self.config.alpha_max,
            shape=shape,
            rng=rng,
        )

        return distortion.EllipseStreakConfig(
            thickness=thickness,
            aspect_ratio=aspect_ratio,
            short_side_min=short_side_min,
            short_side_step=short_side_step,
            alpha=alpha,
        )


ellipse_streak_policy_factory = DistortionPolicyFactory(
    distortion.ellipse_streak,
    EllipseStreakConfigGenerator,
)
