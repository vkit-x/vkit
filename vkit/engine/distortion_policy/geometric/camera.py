from typing import Tuple

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np

from vkit.engine import distortion
from ..type import DistortionConfigGenerator, DistortionPolicyFactory
from ..opt import sample_int, sample_float, generate_grid_size


def sample_camera_model_config(
    level: int,
    level_1_max: int,
    rotation_theta_max: int,
    rng: RandomGenerator,
):
    rotation_theta = sample_int(
        level=level,
        value_min=1,
        value_max=rotation_theta_max,
        prob_negative=0.5,
        rng=rng,
    )

    theta_xy = rng.uniform(0, 2 * np.pi)
    vec_x = np.cos(theta_xy)
    vec_y = np.sin(theta_xy)
    vec_z = 0.0

    if level > level_1_max:
        vec_z = rng.uniform(0, 1)
        # NOTE: will be normalized to unit vector in CameraModel.prep_rotation_unit_vec.
        vec_x = (1 - vec_z) * vec_x
        vec_y = (1 - vec_z) * vec_y

    return distortion.CameraModelConfig(
        rotation_unit_vec=[vec_x, vec_y, vec_z],
        rotation_theta=rotation_theta,
    )


@attrs.define
class CameraPlaneOnlyConfigGeneratorConfig:
    level_1_max: int = 5
    rotation_theta_max: int = 40
    grid_size_min: int = 15
    grid_size_ratio: float = 0.01


class CameraPlaneOnlyConfigGenerator(
    DistortionConfigGenerator[
        CameraPlaneOnlyConfigGeneratorConfig,
        distortion.CameraPlaneOnlyConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        camera_model_config = sample_camera_model_config(
            level=self.level,
            level_1_max=self.config.level_1_max,
            rotation_theta_max=self.config.rotation_theta_max,
            rng=rng,
        )
        grid_size = generate_grid_size(
            self.config.grid_size_min,
            self.config.grid_size_ratio,
            shape,
        )
        return distortion.CameraPlaneOnlyConfig(
            camera_model_config=camera_model_config,
            grid_size=grid_size,
        )


camera_plane_only_policy_factory = DistortionPolicyFactory(
    distortion.camera_plane_only,
    CameraPlaneOnlyConfigGenerator,
)


@attrs.define
class CameraCubicCurveConfigGeneratorConfig:
    curve_slope_range_min: float = 10.0
    curve_slope_range_max: float = 90.0
    curve_slope_max: float = 45
    level_1_max: int = 5
    rotation_theta_max: int = 40
    grid_size_min: int = 15
    grid_size_ratio: float = 0.01


class CameraCubicCurveConfigGenerator(
    DistortionConfigGenerator[
        CameraCubicCurveConfigGeneratorConfig,
        distortion.CameraCubicCurveConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        curve_slope_range = sample_float(
            level=self.level,
            value_min=self.config.curve_slope_range_min,
            value_max=self.config.curve_slope_range_max,
            prob_reciprocal=None,
            rng=rng,
        )
        alpha_ratio = rng.uniform()
        curve_alpha = curve_slope_range * alpha_ratio
        curve_beta = curve_slope_range - curve_alpha

        # Clip.
        curve_alpha = min(self.config.curve_slope_max, curve_alpha)
        curve_beta = min(self.config.curve_slope_max, curve_beta)

        if rng.random() < 0.5:
            curve_alpha *= -1
        if rng.random() < 0.5:
            curve_beta *= -1

        curve_direction = rng.uniform(0, 180)

        camera_model_config = sample_camera_model_config(
            level=self.level,
            level_1_max=self.config.level_1_max,
            rotation_theta_max=self.config.rotation_theta_max,
            rng=rng,
        )
        grid_size = generate_grid_size(
            self.config.grid_size_min,
            self.config.grid_size_ratio,
            shape,
        )
        return distortion.CameraCubicCurveConfig(
            curve_alpha=curve_alpha,
            curve_beta=curve_beta,
            curve_direction=curve_direction,
            curve_scale=1.0,
            camera_model_config=camera_model_config,
            grid_size=grid_size,
        )


camera_cubic_curve_policy_factory = DistortionPolicyFactory(
    distortion.camera_cubic_curve,
    CameraCubicCurveConfigGenerator,
)


@attrs.define
class CameraPlaneLineFoldConfigGeneratorConfig:
    fold_alpha_min: float = 0.1
    fold_alpha_max: float = 1.25
    level_1_max: int = 5
    rotation_theta_max: int = 40
    grid_size_min: int = 15
    grid_size_ratio: float = 0.01


class CameraPlaneLineFoldConfigGenerator(
    DistortionConfigGenerator[
        CameraPlaneLineFoldConfigGeneratorConfig,
        distortion.CameraPlaneLineFoldConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        height, width = shape
        fold_point = (rng.integers(0, width), rng.integers(0, height))

        fold_direction = rng.uniform(0, 180)

        fold_perturb_vec_z = max(shape) / 4
        if rng.random() < 0.5:
            fold_perturb_vec_z *= -1.0
        fold_perturb_vec = (0.0, 0.0, fold_perturb_vec_z)

        fold_alpha = sample_float(
            level=self.level,
            value_min=self.config.fold_alpha_min,
            value_max=self.config.fold_alpha_max,
            prob_reciprocal=None,
            rng=rng,
            inverse_level=True,
        )

        camera_model_config = sample_camera_model_config(
            level=self.level,
            level_1_max=self.config.level_1_max,
            rotation_theta_max=self.config.rotation_theta_max,
            rng=rng,
        )
        grid_size = generate_grid_size(
            self.config.grid_size_min,
            self.config.grid_size_ratio,
            shape,
        )
        return distortion.CameraPlaneLineFoldConfig(
            fold_point=fold_point,
            fold_direction=fold_direction,
            fold_perturb_vec=fold_perturb_vec,
            fold_alpha=fold_alpha,
            camera_model_config=camera_model_config,
            grid_size=grid_size,
        )


camera_plane_line_fold_policy_factory = DistortionPolicyFactory(
    distortion.camera_plane_line_fold,
    CameraPlaneLineFoldConfigGenerator,
)


@attrs.define
class CameraPlaneLineCurveConfigGeneratorConfig:
    curve_alpha_min: float = 1.0
    curve_alpha_max: float = 2.0
    level_1_max: int = 5
    rotation_theta_max: int = 40
    grid_size_min: int = 15
    grid_size_ratio: float = 0.01


class CameraPlaneLineCurveConfigGenerator(
    DistortionConfigGenerator[
        CameraPlaneLineCurveConfigGeneratorConfig,
        distortion.CameraPlaneLineCurveConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        height, width = shape
        curve_point = (rng.integers(0, width), rng.integers(0, height))

        curve_direction = rng.uniform(0, 180)

        curve_perturb_vec_z = max(shape) / 4
        if rng.random() < 0.5:
            curve_perturb_vec_z *= -1.0
        curve_perturb_vec = (0.0, 0.0, curve_perturb_vec_z)

        curve_alpha = sample_float(
            level=self.level,
            value_min=self.config.curve_alpha_min,
            value_max=self.config.curve_alpha_max,
            prob_reciprocal=None,
            rng=rng,
            inverse_level=True,
        )

        camera_model_config = sample_camera_model_config(
            level=self.level,
            level_1_max=self.config.level_1_max,
            rotation_theta_max=self.config.rotation_theta_max,
            rng=rng,
        )
        grid_size = generate_grid_size(
            self.config.grid_size_min,
            self.config.grid_size_ratio,
            shape,
        )
        return distortion.CameraPlaneLineCurveConfig(
            curve_point=curve_point,
            curve_direction=curve_direction,
            curve_perturb_vec=curve_perturb_vec,
            curve_alpha=curve_alpha,
            camera_model_config=camera_model_config,
            grid_size=grid_size,
        )


camera_plane_line_curve_policy_factory = DistortionPolicyFactory(
    distortion.camera_plane_line_curve,
    CameraPlaneLineCurveConfigGenerator,
)
