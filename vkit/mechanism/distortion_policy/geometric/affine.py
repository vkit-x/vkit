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
from typing import Tuple

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.mechanism import distortion
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
    angle_min: int = 1
    angle_max: int = 180
    prob_negative: float = 0.5


class RotateConfigGenerator(
    DistortionConfigGenerator[
        RotateConfigGeneratorConfig,
        distortion.RotateConfig,
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
