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
class GaussianBlurConfigGeneratorConfig:
    sigma_min: float = 0.5
    sigma_max: float = 1.0


class GaussianBlurConfigGenerator(
    DistortionConfigGenerator[
        GaussianBlurConfigGeneratorConfig,
        distortion.GaussianBlurConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        sigma = sample_float(
            level=self.level,
            value_min=self.config.sigma_min,
            value_max=self.config.sigma_max,
            prob_reciprocal=None,
            rng=rng,
        )

        return distortion.GaussianBlurConfig(sigma=sigma)


gaussian_blur_policy_factory = DistortionPolicyFactory(
    distortion.gaussian_blur,
    GaussianBlurConfigGenerator,
)


@attrs.define
class DefocusBlurConfigGeneratorConfig:
    radius_min: int = 1
    radius_max: int = 2


class DefocusBlurConfigGenerator(
    DistortionConfigGenerator[
        DefocusBlurConfigGeneratorConfig,
        distortion.DefocusBlurConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        radius = sample_int(
            level=self.level,
            value_min=self.config.radius_min,
            value_max=self.config.radius_max,
            prob_negative=None,
            rng=rng,
        )

        return distortion.DefocusBlurConfig(radius=radius)


defocus_blur_policy_factory = DistortionPolicyFactory(
    distortion.defocus_blur,
    DefocusBlurConfigGenerator,
)


@attrs.define
class MotionBlurConfigGeneratorConfig:
    radius_min: int = 1
    radius_max: int = 2


class MotionBlurConfigGenerator(
    DistortionConfigGenerator[
        MotionBlurConfigGeneratorConfig,
        distortion.MotionBlurConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        radius = sample_int(
            level=self.level,
            value_min=self.config.radius_min,
            value_max=self.config.radius_max,
            prob_negative=None,
            rng=rng,
        )
        angle = rng.integers(0, 360)

        return distortion.MotionBlurConfig(
            radius=radius,
            angle=angle,
        )


motion_blur_policy_factory = DistortionPolicyFactory(
    distortion.motion_blur,
    MotionBlurConfigGenerator,
)


@attrs.define
class GlassBlurConfigGeneratorConfig:
    sigma_min: float = 0.5
    sigma_max: float = 1.0
    delta_min: int = 1
    delta_max: int = 1
    loop_min: int = 1
    loop_max: int = 4


class GlassBlurConfigGenerator(
    DistortionConfigGenerator[
        GlassBlurConfigGeneratorConfig,
        distortion.GlassBlurConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        sigma = sample_float(
            level=self.level,
            value_min=self.config.sigma_min,
            value_max=self.config.sigma_max,
            prob_reciprocal=None,
            rng=rng,
        )
        delta = sample_int(
            level=self.level,
            value_min=self.config.delta_min,
            value_max=self.config.delta_max,
            prob_negative=None,
            rng=rng,
        )
        loop = sample_int(
            level=self.level,
            value_min=self.config.loop_min,
            value_max=self.config.loop_max,
            prob_negative=None,
            rng=rng,
        )

        return distortion.GlassBlurConfig(
            sigma=sigma,
            delta=delta,
            loop=loop,
        )


glass_blur_policy_factory = DistortionPolicyFactory(
    distortion.glass_blur,
    GlassBlurConfigGenerator,
)


@attrs.define
class ZoomInBlurConfigGeneratorConfig:
    ratio_min: float = 0.01
    ratio_max: float = 0.1
    step_min: float = 0.002
    step_max: float = 0.02
    alpha_min: float = 0.5
    alpha_max: float = 0.7


class ZoomInBlurConfigGenerator(
    DistortionConfigGenerator[
        ZoomInBlurConfigGeneratorConfig,
        distortion.ZoomInBlurConfig,
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
        step = sample_float(
            level=self.level,
            value_min=self.config.step_min,
            value_max=self.config.step_max,
            prob_reciprocal=None,
            rng=rng,
        )
        alpha = rng.uniform(self.config.alpha_min, self.config.alpha_max)

        return distortion.ZoomInBlurConfig(
            ratio=ratio,
            step=step,
            alpha=alpha,
        )


zoom_in_blur_policy_factory = DistortionPolicyFactory(
    distortion.zoom_in_blur,
    ZoomInBlurConfigGenerator,
)
