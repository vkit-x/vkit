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
class JpegQualityConfigGeneratorConfig:
    quality_min: int = 1
    quality_max: int = 50


class JpegQualityConfigGenerator(
    DistortionConfigGenerator[
        JpegQualityConfigGeneratorConfig,
        distortion.JpegQualityConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        quality = sample_int(
            level=self.level,
            value_min=self.config.quality_min,
            value_max=self.config.quality_max,
            prob_negative=None,
            rng=rng,
            inverse_level=True
        )

        return distortion.JpegQualityConfig(quality=quality)


jpeg_quality_policy_factory = DistortionPolicyFactory(
    distortion.jpeg_quality,
    JpegQualityConfigGenerator,
)


@attrs.define
class PixelationConfigGeneratorConfig:
    ratio_min: float = 0.3
    ratio_max: float = 1.0


class PixelationConfigGenerator(
    DistortionConfigGenerator[
        PixelationConfigGeneratorConfig,
        distortion.PixelationConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        ratio = sample_float(
            level=self.level,
            value_min=self.config.ratio_min,
            value_max=self.config.ratio_max,
            prob_reciprocal=None,
            rng=rng,
            inverse_level=True,
        )

        return distortion.PixelationConfig(ratio=ratio)


pixelation_policy_factory = DistortionPolicyFactory(
    distortion.pixelation,
    PixelationConfigGenerator,
)


@attrs.define
class FogConfigGeneratorConfig:
    roughness_min: float = 0.2
    roughness_max: float = 0.85
    ratio_max_min: float = 0.2
    ratio_max_max: float = 0.75


class FogConfigGenerator(
    DistortionConfigGenerator[
        FogConfigGeneratorConfig,
        distortion.FogConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        roughness = sample_float(
            level=self.level,
            value_min=self.config.roughness_min,
            value_max=self.config.roughness_max,
            prob_reciprocal=None,
            rng=rng,
        )
        ratio_max = sample_float(
            level=self.level,
            value_min=self.config.ratio_max_min,
            value_max=self.config.ratio_max_max,
            prob_reciprocal=None,
            rng=rng,
        )

        return distortion.FogConfig(
            roughness=roughness,
            ratio_max=ratio_max,
        )


fog_policy_factory = DistortionPolicyFactory(
    distortion.fog,
    FogConfigGenerator,
)
