from typing import Tuple

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.engine import distortion
from ..type import DistortionConfigGenerator, DistortionPolicyFactory
from ..opt import sample_int, sample_float


@attrs.define
class ShearHoriConfigGeneratorConfig:
    angle_min: int = 1
    angle_max: int = 30
    prob_negative: float = 0.5


class ShearHoriConfigGenerator(
    DistortionConfigGenerator[
        ShearHoriConfigGeneratorConfig,
        distortion.ShearHoriConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        angle = sample_int(
            level=self.level,
            value_min=self.config.angle_min,
            value_max=self.config.angle_max,
            prob_negative=self.config.prob_negative,
            rng=rng,
        )

        return distortion.ShearHoriConfig(angle=angle)


shear_hori_policy_factory = DistortionPolicyFactory(
    distortion.shear_hori,
    ShearHoriConfigGenerator,
)


@attrs.define
class ShearVertConfigGeneratorConfig:
    angle_min: int = 1
    angle_max: int = 30
    prob_negative: float = 0.5


class ShearVertConfigGenerator(
    DistortionConfigGenerator[
        ShearVertConfigGeneratorConfig,
        distortion.ShearVertConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        angle = sample_int(
            level=self.level,
            value_min=self.config.angle_min,
            value_max=self.config.angle_max,
            prob_negative=self.config.prob_negative,
            rng=rng,
        )

        return distortion.ShearVertConfig(angle=angle)


shear_vert_policy_factory = DistortionPolicyFactory(
    distortion.shear_vert,
    ShearVertConfigGenerator,
)


@attrs.define
class RotateConfigGeneratorConfig:
    level_1_max: int = 6
    level_1_angle_min: int = 1
    level_1_angle_max: int = 45

    level_2_max: int = 8
    level_2_angle_min: int = 46
    level_2_angle_max: int = 90

    level_3_angle_min: int = 91
    level_3_angle_max: int = 180

    prob_negative: float = 0.5


class RotateConfigGenerator(
    DistortionConfigGenerator[
        RotateConfigGeneratorConfig,
        distortion.RotateConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        if self.level <= self.config.level_1_max:
            angle = sample_int(
                level=self.level,
                value_min=self.config.level_1_angle_min,
                value_max=self.config.level_1_angle_max,
                prob_negative=self.config.prob_negative,
                rng=rng,
            )

        elif self.level <= self.config.level_2_max:
            angle = sample_int(
                level=self.level,
                value_min=self.config.level_2_angle_min,
                value_max=self.config.level_2_angle_max,
                prob_negative=self.config.prob_negative,
                rng=rng,
            )

        else:
            angle = sample_int(
                level=self.level,
                value_min=self.config.level_3_angle_min,
                value_max=self.config.level_3_angle_max,
                prob_negative=self.config.prob_negative,
                rng=rng,
            )

        return distortion.RotateConfig(angle=angle)


rotate_policy_factory = DistortionPolicyFactory(
    distortion.rotate,
    RotateConfigGenerator,
)


@attrs.define
class SkewHoriConfigGeneratorConfig:
    ratio_min: float = 0.0
    ratio_max: float = 0.35
    prob_negative: float = 0.5


class SkewHoriConfigGenerator(
    DistortionConfigGenerator[
        SkewHoriConfigGeneratorConfig,
        distortion.SkewHoriConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        ratio = sample_float(
            level=self.level,
            value_min=self.config.ratio_min,
            value_max=self.config.ratio_max,
            prob_reciprocal=None,
            rng=rng,
        )
        if rng.random() < self.config.prob_negative:
            ratio *= -1

        return distortion.SkewHoriConfig(ratio=ratio)


skew_hori_policy_factory = DistortionPolicyFactory(
    distortion.skew_hori,
    SkewHoriConfigGenerator,
)


@attrs.define
class SkewVertConfigGeneratorConfig:
    ratio_min: float = 0.0
    ratio_max: float = 0.35
    prob_negative: float = 0.5


class SkewVertConfigGenerator(
    DistortionConfigGenerator[
        SkewVertConfigGeneratorConfig,
        distortion.SkewVertConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        ratio = sample_float(
            level=self.level,
            value_min=self.config.ratio_min,
            value_max=self.config.ratio_max,
            prob_reciprocal=None,
            rng=rng,
        )
        if rng.random() < self.config.prob_negative:
            ratio *= -1

        return distortion.SkewVertConfig(ratio=ratio)


skew_vert_policy_factory = DistortionPolicyFactory(
    distortion.skew_vert,
    SkewVertConfigGenerator,
)
