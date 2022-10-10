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
from typing import (
    cast,
    Callable,
    Sequence,
    Optional,
    Tuple,
    Union,
    Iterable,
    TypeVar,
)
import math

import attrs
import cv2 as cv
import numpy as np
from numpy.random import Generator as RandomGenerator

from vkit.element import Point, PointList, PointTuple
from ..interface import DistortionConfig
from .grid_rendering.interface import (
    PointProjector,
    DistortionStateImageGridBased,
    DistortionImageGridBased,
)
from .grid_rendering.grid_creator import create_src_image_grid

_T_CONFIG = TypeVar('_T_CONFIG', bound=DistortionConfig)


class Point2dTo3dStrategy:

    def generate_np_3d_points(self, points: PointTuple) -> np.ndarray:
        raise NotImplementedError()


@attrs.define
class CameraModelConfig:
    rotation_unit_vec: Sequence[float]
    rotation_theta: float
    focal_length: Optional[float] = None
    principal_point: Optional[Sequence[float]] = None
    camera_distance: Optional[float] = None


class CameraModel:

    @classmethod
    def prep_rotation_unit_vec(cls, rotation_unit_vec: Sequence[float]) -> np.ndarray:
        vec = np.asarray(rotation_unit_vec, dtype=np.float32)
        length = np.linalg.norm(vec)
        if length != 1.0:
            vec /= length
        return vec

    @classmethod
    def prep_rotation_theta(cls, rotation_theta: float):
        return float(np.clip(rotation_theta, -89, 89) / 180 * np.pi)

    @classmethod
    def prep_principal_point(cls, principal_point: Sequence[float]):
        principal_point = list(principal_point)
        if len(principal_point) == 2:
            principal_point.append(0)
        return np.asarray(principal_point, dtype=np.float32).reshape(-1, 1)

    @classmethod
    def generate_rotation_vec(cls, rotation_unit_vec: np.ndarray, rotation_theta: float):
        # Defines how the object is rotated, following the right hand rule:
        # https://en.wikipedia.org/wiki/Right-hand_rule#Rotations
        #
        # Multiplication is for cv.Rodrigues, as explained in
        # https://stackoverflow.com/a/12977143
        return rotation_unit_vec * rotation_theta

    @classmethod
    def generate_rotation_mat_and_translation_vec(
        cls,
        rotation_vec: np.ndarray,
        camera_distance: float,
        principal_point: np.ndarray,
    ):
        # rotation_mat is a 3x3 matrix [rotated-x, rotated-y, rotated-z]
        rotation_mat, _ = cv.Rodrigues(rotation_vec)
        rotation_mat = cast(np.ndarray, rotation_mat)

        # Assume that image is place in the world coordinate in the way as followed:
        # 1. top-left corner: (0, 0, *)
        # 2. top-right corner: (width - 1, 0, *)
        # 3. bottom-left corner: (0, height - 1, *)
        # 4. bottom-right corner: (width - 1, height - 1, *)
        #
        # principal_point is defined to represent the intersection between axis-z
        # of the camera coordinate and the image. principal_point is defined in the
        # world coordinate with axis-z = 0. principal_point should be in position
        # (0, 0, camera_distance) after transformation.
        #
        # "cc_" is for camera coordinate and "wc_" is for world coordinate.
        cc_principal_point_vec = np.asarray(
            [0, 0, camera_distance],
            dtype=np.float32,
        ).reshape(-1, 1)
        wc_shifted_origin_vec = np.matmul(
            # NOTE: The rotation matrix is orthogonal,
            # hence it's inverse is equal to it's transpose.
            rotation_mat.transpose(),
            cc_principal_point_vec,
        )
        wc_shifted_principal_point_vec = wc_shifted_origin_vec - principal_point

        translation_vec: np.ndarray = np.matmul(
            rotation_mat,
            wc_shifted_principal_point_vec.reshape(-1, 1),
        )
        return rotation_mat, translation_vec

    @classmethod
    def generate_translation_vec(
        cls,
        rotation_vec: np.ndarray,
        camera_distance: float,
        principal_point: np.ndarray,
    ):
        _, translation_vec = CameraModel.generate_rotation_mat_and_translation_vec(
            rotation_vec,
            camera_distance,
            principal_point,
        )
        return translation_vec

    @classmethod
    def generate_extrinsic_mat(
        cls,
        rotation_unit_vec: np.ndarray,
        rotation_theta: float,
        camera_distance: float,
        principal_point: np.ndarray,
    ):
        rotation_vec = CameraModel.generate_rotation_vec(rotation_unit_vec, rotation_theta)
        rotation_mat, translation_vec = CameraModel.generate_rotation_mat_and_translation_vec(
            rotation_vec,
            camera_distance,
            principal_point,
        )
        extrinsic_mat = np.hstack((rotation_mat, translation_vec.reshape((-1, 1))))
        return extrinsic_mat

    @classmethod
    def generate_intrinsic_mat(cls, focal_length: float):
        return np.asarray(
            [
                [focal_length, 0, 0],
                [0, focal_length, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def __init__(self, config: CameraModelConfig):
        assert config.focal_length
        assert config.camera_distance
        assert config.principal_point

        rotation_unit_vec = self.prep_rotation_unit_vec(config.rotation_unit_vec)
        rotation_theta = self.prep_rotation_theta(config.rotation_theta)
        self.rotation_vec = self.generate_rotation_vec(rotation_unit_vec, rotation_theta)

        principal_point = self.prep_principal_point(config.principal_point)
        self.translation_vec = self.generate_translation_vec(
            self.rotation_vec,
            config.camera_distance,
            principal_point,
        )
        self.intrinsic_mat = self.generate_intrinsic_mat(config.focal_length)

    def project_np_points_from_3d_to_2d(self, np_3d_points: np.ndarray) -> np.ndarray:
        camera_2d_points, _ = cv.projectPoints(
            np_3d_points,
            self.rotation_vec,
            self.translation_vec,
            self.intrinsic_mat,
            np.zeros(5),
        )
        return camera_2d_points.reshape(-1, 2)


class CameraPointProjector(PointProjector):

    def __init__(
        self,
        point_2d_to_3d_strategy: Point2dTo3dStrategy,
        camera_model_config: CameraModelConfig,
    ):
        self.point_2d_to_3d_strategy = point_2d_to_3d_strategy
        self.camera_model = CameraModel(camera_model_config)

    def project_points(self, src_points: Union[PointList, PointTuple, Iterable[Point]]):
        np_3d_points = self.point_2d_to_3d_strategy.generate_np_3d_points(PointTuple(src_points))
        camera_2d_points = self.camera_model.project_np_points_from_3d_to_2d(np_3d_points)
        return PointTuple.from_np_array(camera_2d_points)

    def project_point(self, src_point: Point):
        return self.project_points(PointTuple.from_point(src_point))[0]


class DistortionStateCameraOperation(DistortionStateImageGridBased[_T_CONFIG]):

    @classmethod
    def complete_camera_model_config(
        cls,
        height: int,
        width: int,
        camera_model_config: CameraModelConfig,
    ):
        if camera_model_config.principal_point \
                and camera_model_config.focal_length \
                and camera_model_config.camera_distance:
            return camera_model_config

        # Make a copy.
        camera_model_config = attrs.evolve(camera_model_config)

        if not camera_model_config.principal_point:
            camera_model_config.principal_point = [height // 2, width // 2]

        if not camera_model_config.focal_length \
                or not camera_model_config.camera_distance:
            camera_model_config.focal_length = max(height, width)
            camera_model_config.camera_distance = camera_model_config.focal_length

        return camera_model_config

    def initialize_camera_operation(
        self,
        height: int,
        width: int,
        grid_size: int,
        point_2d_to_3d_strategy: Point2dTo3dStrategy,
        camera_model_config: CameraModelConfig,
    ):
        src_image_grid = create_src_image_grid(height, width, grid_size)

        camera_model_config = self.complete_camera_model_config(
            height,
            width,
            camera_model_config,
        )
        point_projector = CameraPointProjector(
            point_2d_to_3d_strategy,
            camera_model_config,
        )

        self.initialize_image_grid_based(src_image_grid, point_projector)


@attrs.define
class CameraPlaneOnlyConfig(DistortionConfig):
    camera_model_config: CameraModelConfig
    grid_size: int


class CameraPlaneOnlyPoint2dTo3dStrategy(Point2dTo3dStrategy):

    def generate_np_3d_points(self, points: PointTuple) -> np.ndarray:
        np_2d_points = points.to_smooth_np_array()
        np_3d_points = np.hstack((
            np_2d_points,
            np.zeros((np_2d_points.shape[0], 1), dtype=np.float32),
        ))
        return np_3d_points


class CameraPlaneOnlyState(DistortionStateCameraOperation[CameraPlaneOnlyConfig]):

    @classmethod
    def weights_func(cls, norm_distances: np.ndarray, alpha: float):
        return alpha / (norm_distances + alpha)

    def __init__(
        self,
        config: CameraPlaneOnlyConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        height, width = shape
        self.initialize_camera_operation(
            height,
            width,
            config.grid_size,
            CameraPlaneOnlyPoint2dTo3dStrategy(),
            config.camera_model_config,
        )


camera_plane_only = DistortionImageGridBased(
    config_cls=CameraPlaneOnlyConfig,
    state_cls=CameraPlaneOnlyState,
)


@attrs.define
class CameraCubicCurveConfig(DistortionConfig):
    curve_alpha: float
    curve_beta: float
    # Clockwise, [0, 180]
    curve_direction: float
    curve_scale: float
    camera_model_config: CameraModelConfig
    grid_size: int


class CameraCubicCurvePoint2dTo3dStrategy(Point2dTo3dStrategy):

    def __init__(
        self,
        height: int,
        width: int,
        curve_alpha: float,
        curve_beta: float,
        curve_direction: float,
        curve_scale: float,
    ):
        # Plane area.
        self.height = height
        self.width = width

        # Curve endpoint slopes.
        self.curve_alpha = math.tan(np.clip(curve_alpha, -80, 80) / 180 * np.pi)
        self.curve_beta = math.tan(np.clip(curve_beta, -80, 80) / 180 * np.pi)

        # Plane projection direction.
        self.curve_direction = (curve_direction % 180) / 180 * np.pi

        self.rotation_mat = np.asarray(
            [
                [
                    math.cos(self.curve_direction),
                    math.sin(self.curve_direction),
                ],
                [
                    -math.sin(self.curve_direction),
                    math.cos(self.curve_direction),
                ],
            ],
            dtype=np.float32,
        )

        corners = np.asarray(
            [
                [0, 0],
                [self.width - 1, 0],
                [self.width - 1, self.height - 1],
                [0, self.height - 1],
            ],
            dtype=np.float32,
        )
        rotated_corners = np.matmul(self.rotation_mat, corners.transpose())
        self.plane_projection_min = rotated_corners[0].min()
        self.plane_projection_range = rotated_corners[0].max() - self.plane_projection_min

        self.curve_scale = curve_scale

    def generate_np_3d_points(self, points: PointTuple) -> np.ndarray:
        np_2d_points = points.to_smooth_np_array()

        # Project based on theta.
        plane_projected_points = np.matmul(self.rotation_mat, np_2d_points.transpose())
        plane_projected_xs = plane_projected_points[0]
        plane_projected_ratios = (
            plane_projected_xs - self.plane_projection_min
        ) / self.plane_projection_range

        # Axis-z.
        poly = np.asarray([
            self.curve_alpha + self.curve_beta,
            -2 * self.curve_alpha - self.curve_beta,
            self.curve_alpha,
            0,
        ])
        pos_zs = np.polyval(poly, plane_projected_ratios)
        pos_zs = pos_zs * self.plane_projection_range * self.curve_scale
        # Shift mean to zero.
        pos_zs: np.ndarray = pos_zs - pos_zs.mean()

        np_3d_points = np.hstack((np_2d_points, pos_zs.reshape((-1, 1))))
        return np_3d_points


class CameraCubicCurveState(DistortionStateCameraOperation[CameraCubicCurveConfig]):

    def __init__(
        self,
        config: CameraCubicCurveConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        height, width = shape
        self.initialize_camera_operation(
            height,
            width,
            config.grid_size,
            CameraCubicCurvePoint2dTo3dStrategy(
                height,
                width,
                config.curve_alpha,
                config.curve_beta,
                config.curve_direction,
                config.curve_scale,
            ),
            config.camera_model_config,
        )


camera_cubic_curve = DistortionImageGridBased(
    config_cls=CameraCubicCurveConfig,
    state_cls=CameraCubicCurveState,
)


class CameraPlaneLinePoint2dTo3dStrategy(Point2dTo3dStrategy):

    def __init__(
        self,
        height: int,
        width: int,
        point: Tuple[float, float],
        direction: float,
        perturb_vec: Tuple[float, float, float],
        alpha: float,
        weights_func: Callable[[np.ndarray, float], np.ndarray],
    ):
        # Plane area.
        self.height = height
        self.width = width

        # Define a line.
        self.point = np.asarray(point, dtype=np.float32)
        direction = (direction % 180) / 180 * np.pi
        cos_theta = np.cos(direction)
        sin_theta = np.sin(direction)
        self.line_params_a_b = np.asarray([sin_theta, -cos_theta], dtype=np.float32)
        self.line_param_c = -self.point[0] * sin_theta + self.point[1] * cos_theta

        # For weight calculationn.
        self.distance_max = np.sqrt(height**2 + width**2)
        self.alpha = alpha
        self.weights_func = weights_func

        # Deformation vector.
        self.perturb_vec = np.asarray(perturb_vec, dtype=np.float32)

    def generate_np_3d_points(self, points: PointTuple) -> np.ndarray:
        np_2d_points = points.to_smooth_np_array()

        # Calculate weights.
        distances = np.abs((np_2d_points * self.line_params_a_b).sum(axis=1) + self.line_param_c)
        norm_distances = distances / self.distance_max
        weights = self.weights_func(norm_distances, self.alpha)

        # Add weighted fold vector.
        np_3d_points = np.hstack(
            (np_2d_points, np.zeros((np_2d_points.shape[0], 1), dtype=np.float32))
        )
        np_perturb = weights.reshape(-1, 1) * self.perturb_vec
        # Shift mean to zero.
        np_perturb -= np_perturb.mean(axis=0)
        np_3d_points += np_perturb
        return np_3d_points


@attrs.define
class CameraPlaneLineFoldConfig(DistortionConfig):
    fold_point: Tuple[float, float]
    # Clockwise, [0, 180]
    fold_direction: float
    fold_perturb_vec: Tuple[float, float, float]
    fold_alpha: float
    camera_model_config: CameraModelConfig
    grid_size: int


class CameraPlaneLineFoldState(DistortionStateCameraOperation[CameraPlaneLineFoldConfig]):

    @classmethod
    def weights_func(cls, norm_distances: np.ndarray, alpha: float):
        return alpha / (norm_distances + alpha)

    def __init__(
        self,
        config: CameraPlaneLineFoldConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        height, width = shape
        self.initialize_camera_operation(
            height,
            width,
            config.grid_size,
            CameraPlaneLinePoint2dTo3dStrategy(
                height=height,
                width=width,
                point=config.fold_point,
                direction=config.fold_direction,
                perturb_vec=config.fold_perturb_vec,
                alpha=config.fold_alpha,
                weights_func=self.weights_func,
            ),
            config.camera_model_config,
        )


camera_plane_line_fold = DistortionImageGridBased(
    config_cls=CameraPlaneLineFoldConfig,
    state_cls=CameraPlaneLineFoldState,
)


@attrs.define
class CameraPlaneLineCurveConfig(DistortionConfig):
    curve_point: Tuple[float, float]
    # Clockwise, [0, 180]
    curve_direction: float
    curve_perturb_vec: Tuple[float, float, float]
    curve_alpha: float
    camera_model_config: CameraModelConfig
    grid_size: int


class CameraPlaneLineCurveState(DistortionStateCameraOperation[CameraPlaneLineCurveConfig]):

    @classmethod
    def weights_func(cls, norm_distances: np.ndarray, alpha: float):
        return 1 - norm_distances**alpha

    def __init__(
        self,
        config: CameraPlaneLineCurveConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        height, width = shape
        self.initialize_camera_operation(
            height,
            width,
            config.grid_size,
            CameraPlaneLinePoint2dTo3dStrategy(
                height=height,
                width=width,
                point=config.curve_point,
                direction=config.curve_direction,
                perturb_vec=config.curve_perturb_vec,
                alpha=config.curve_alpha,
                weights_func=self.weights_func,
            ),
            config.camera_model_config,
        )


camera_plane_line_curve = DistortionImageGridBased(
    config_cls=CameraPlaneLineCurveConfig,
    state_cls=CameraPlaneLineCurveState,
)
