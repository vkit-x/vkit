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
from ..opt import sample_float


@attrs.define
class GaussionNoiseConfigGeneratorConfig:
    std_min: float = 1.0
    std_max: float = 35.0


class GaussionNoiseConfigGenerator(
    DistortionConfigGenerator[
        GaussionNoiseConfigGeneratorConfig,
        distortion.GaussionNoiseConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        std = sample_float(
            level=self.level,
            value_min=self.config.std_min,
            value_max=self.config.std_max,
            prob_reciprocal=None,
            rng=rng,
        )

        return distortion.GaussionNoiseConfig(std=std)


gaussion_noise_policy_factory = DistortionPolicyFactory(
    distortion.gaussion_noise,
    GaussionNoiseConfigGenerator,
)


@attrs.define
class PoissonNoiseConfigGeneratorConfig:
    pass


class PoissonNoiseConfigGenerator(
    DistortionConfigGenerator[
        PoissonNoiseConfigGeneratorConfig,
        distortion.PoissonNoiseConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        return distortion.PoissonNoiseConfig()


poisson_noise_policy_factory = DistortionPolicyFactory(
    distortion.poisson_noise,
    PoissonNoiseConfigGenerator,
)


@attrs.define
class ImpulseNoiseConfigGeneratorConfig:
    prob_presv_min: float = 0.95
    prob_presv_max: float = 1.0


class ImpulseNoiseConfigGenerator(
    DistortionConfigGenerator[
        ImpulseNoiseConfigGeneratorConfig,
        distortion.ImpulseNoiseConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        prob_presv = sample_float(
            level=self.level,
            value_min=self.config.prob_presv_min,
            value_max=self.config.prob_presv_max,
            prob_reciprocal=None,
            rng=rng,
            inverse_level=True,
        )
        prob_not_presv = 1 - prob_presv

        salt_ratio = rng.uniform()
        prob_salt = prob_not_presv * salt_ratio
        prob_pepper = prob_not_presv - prob_salt

        return distortion.ImpulseNoiseConfig(
            prob_salt=prob_salt,
            prob_pepper=prob_pepper,
        )


impulse_noise_policy_factory = DistortionPolicyFactory(
    distortion.impulse_noise,
    ImpulseNoiseConfigGenerator,
)


@attrs.define
class SpeckleNoiseConfigGeneratorConfig:
    std_min: float = 0.0
    std_max: float = 0.3


class SpeckleNoiseConfigGenerator(
    DistortionConfigGenerator[
        SpeckleNoiseConfigGeneratorConfig,
        distortion.SpeckleNoiseConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        std = sample_float(
            level=self.level,
            value_min=self.config.std_min,
            value_max=self.config.std_max,
            prob_reciprocal=None,
            rng=rng,
        )

        return distortion.SpeckleNoiseConfig(std=std)


speckle_noise_policy_factory = DistortionPolicyFactory(
    distortion.speckle_noise,
    SpeckleNoiseConfigGenerator,
)
