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
from typing import Tuple, Sequence, List, Optional
from enum import Enum, unique
import math
import logging

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv

from vkit.utility import normalize_to_probs
from vkit.element import Point, PointList, Polygon, ScoreMap
from vkit.engine.distortion.geometric.affine import affine_points
from ..interface import PipelineStep, PipelineStepFactory
from .page_text_region import PageTextRegionStepOutput

logger = logging.getLogger(__name__)


@attrs.define
class PageTextRegionLabelStepConfig:
    gaussian_map_length: int = 45
    # 1 centrod + n deviate points.
    num_deviate_char_regression_labels: int = 3


@attrs.define
class PageTextRegionLabelStepInput:
    page_text_region_step_output: PageTextRegionStepOutput


@unique
class PageCharRegressionLabelTag(Enum):
    CENTROID = 'centroid'
    DEVIATE = 'deviate'


PI = float(np.pi)
TWO_PI = float(2 * np.pi)


@attrs.define
class Vector:
    y: int
    x: int

    _distance: Optional[float] = None
    _theta: Optional[float] = None

    def lazy_post_init(self):
        initialized = (self._distance is not None)
        if initialized:
            return

        self._distance = math.hypot(self.x, self.y)
        self._theta = float(np.arctan2(self.y, self.x)) % TWO_PI

    @property
    def distance(self):
        self.lazy_post_init()
        assert self._distance is not None
        return self._distance

    @property
    def theta(self):
        self.lazy_post_init()
        assert self._theta is not None
        return self._theta

    @staticmethod
    def calculate_theta_delta(
        vector0: 'Vector',
        vector1: 'Vector',
        clockwise: bool = False,
    ):
        theta_delta = (vector1.theta - vector0.theta + PI) % TWO_PI - PI
        if clockwise and theta_delta < 0:
            theta_delta += TWO_PI
        return theta_delta

    def dot(self, other: 'Vector'):
        return self.x * other.x + self.y * other.y


@attrs.define
class PageCharRegressionLabel:
    char_idx: int
    tag: PageCharRegressionLabelTag
    label_point: Point
    up_left: Point
    up_right: Point
    down_right: Point
    down_left: Point

    _up_left_vector: Optional[Vector] = None
    _up_right_vector: Optional[Vector] = None
    _down_right_vector: Optional[Vector] = None
    _down_left_vector: Optional[Vector] = None

    _up_left_to_up_right_angle: Optional[float] = None
    _up_right_to_down_right_angle: Optional[float] = None
    _down_right_to_down_left_angle: Optional[float] = None
    _down_left_to_up_left_angle: Optional[float] = None
    _valid: Optional[bool] = None
    _clockwise_angle_distribution: Optional[Sequence[float]] = None

    def lazy_post_init(self):
        initialized = (self._up_left_vector is not None)
        if initialized:
            return

        self._up_left_vector = Vector(
            y=self.up_left.y - self.label_point.y,
            x=self.up_left.x - self.label_point.x,
        )
        self._up_right_vector = Vector(
            y=self.up_right.y - self.label_point.y,
            x=self.up_right.x - self.label_point.x,
        )
        self._down_right_vector = Vector(
            y=self.down_right.y - self.label_point.y,
            x=self.down_right.x - self.label_point.x,
        )
        self._down_left_vector = Vector(
            y=self.down_left.y - self.label_point.y,
            x=self.down_left.x - self.label_point.x,
        )

        self._up_left_to_up_right_angle = Vector.calculate_theta_delta(
            self._up_left_vector,
            self._up_right_vector,
            clockwise=True,
        )
        self._up_right_to_down_right_angle = Vector.calculate_theta_delta(
            self._up_right_vector,
            self._down_right_vector,
            clockwise=True,
        )
        self._down_right_to_down_left_angle = Vector.calculate_theta_delta(
            self._down_right_vector,
            self._down_left_vector,
            clockwise=True,
        )
        self._down_left_to_up_left_angle = Vector.calculate_theta_delta(
            self._down_left_vector,
            self._up_left_vector,
            clockwise=True,
        )

        sum_of_angles = sum([
            self._up_left_to_up_right_angle,
            self._up_right_to_down_right_angle,
            self._down_right_to_down_left_angle,
            self._down_left_to_up_left_angle,
        ])
        # Consider valid if deviate within 4 degrees.
        self._valid = math.isclose(sum_of_angles, TWO_PI, rel_tol=0.012)

        self._clockwise_angle_distribution = normalize_to_probs([
            self._up_left_to_up_right_angle,
            self._up_right_to_down_right_angle,
            self._down_right_to_down_left_angle,
            self._down_left_to_up_left_angle,
        ])

    def to_shifted_page_char_regression_label(self, y_offset: int, x_offset: int):
        assert self.valid

        label_point = self.label_point.to_shifted_point(y_offset=y_offset, x_offset=x_offset)
        up_left = self.up_left.to_shifted_point(y_offset=y_offset, x_offset=x_offset)
        up_right = self.up_right.to_shifted_point(y_offset=y_offset, x_offset=x_offset)
        down_right = self.down_right.to_shifted_point(y_offset=y_offset, x_offset=x_offset)
        down_left = self.down_left.to_shifted_point(y_offset=y_offset, x_offset=x_offset)

        shifted_page_char_regression_label = PageCharRegressionLabel(
            char_idx=self.char_idx,
            tag=self.tag,
            label_point=label_point,
            up_left=up_left,
            up_right=up_right,
            down_right=down_right,
            down_left=down_left,
            # Avoid recalculate the labelings.
            up_left_vector=self._up_left_vector,  # type: ignore
            up_right_vector=self._up_right_vector,  # type: ignore
            down_right_vector=self._down_right_vector,  # type: ignore
            down_left_vector=self._down_left_vector,  # type: ignore
            up_left_to_up_right_angle=self._up_left_to_up_right_angle,  # type: ignore
            up_right_to_down_right_angle=self._up_right_to_down_right_angle,  # type: ignore
            down_right_to_down_left_angle=self._down_right_to_down_left_angle,  # type: ignore
            down_left_to_up_left_angle=self._down_left_to_up_left_angle,  # type: ignore
            valid=self._valid,  # type: ignore
            clockwise_angle_distribution=self._clockwise_angle_distribution,  # type: ignore
        )
        return shifted_page_char_regression_label

    @property
    def valid(self):
        self.lazy_post_init()
        assert self._valid is not None
        return self._valid

    def generate_up_left_offsets(self):
        self.lazy_post_init()
        assert self._up_left_vector is not None
        return self._up_left_vector.y, self._up_left_vector.x

    def generate_clockwise_angle_distribution(self):
        self.lazy_post_init()
        assert self._clockwise_angle_distribution is not None
        return self._clockwise_angle_distribution

    def generate_non_up_left_distances(self):
        self.lazy_post_init()
        assert self._up_right_vector is not None
        assert self._down_right_vector is not None
        assert self._down_left_vector is not None
        return (
            self._up_right_vector.distance,
            self._down_right_vector.distance,
            self._down_left_vector.distance,
        )


@attrs.define
class PageTextRegionLabelStepOutput:
    page_score_map: ScoreMap
    page_char_regression_labels: Sequence[PageCharRegressionLabel]


class PageTextRegionLabelStep(
    PipelineStep[
        PageTextRegionLabelStepConfig,
        PageTextRegionLabelStepInput,
        PageTextRegionLabelStepOutput,
    ]
):  # yapf: disable

    @staticmethod
    def generate_np_gaussian_map(
        length: int,
        rescale_radius_to_value: float = 3.0,
    ):
        # https://colab.research.google.com/drive/1TQ1-BTisMYZHIRVVNpVwDFPviXYMhT7A
        # Build distances to the center point.
        if length % 2 == 0:
            radius = length / 2 - 0.5
        else:
            radius = length // 2

        np_offset = np.abs(np.arange(length, dtype=np.float32) - radius)
        np_vert_offset = np.repeat(np_offset[:, None], length, axis=1)
        np_hori_offset = np.repeat(np_offset[None, :], length, axis=0)
        np_distance = np.sqrt(np.square(np_vert_offset) + np.square(np_hori_offset))

        # Rescale the radius. If default value 3.0 is used, the value lying on the circle of
        # gaussian map is approximately 0.01, and the value in corner is
        # approximately 1E-6.
        np_distance = rescale_radius_to_value * np_distance / radius
        np_gaussian_map = np.exp(-0.5 * np.square(np_distance))

        # For perspective transformation.
        np_points = np.array(
            [
                (0, 0),
                (length - 1, 0),
                (length - 1, length - 1),
                (0, length - 1),
            ],
            dtype=np.float32,
        )

        return np_gaussian_map, np_points

    def generate_page_score_map(
        self,
        shape: Tuple[int, int],
        page_char_polygons: Sequence[Polygon],
    ):
        score_map = ScoreMap.from_shape(shape)

        # Will be transform to fit each char polygon.
        np_src_gaussian_map, np_src_points = self.generate_np_gaussian_map(
            self.config.gaussian_map_length,
        )

        for polygon in page_char_polygons:
            # Transform.
            bounding_box = polygon.fill_np_array_internals.bounding_box
            np_dst_points = polygon.fill_np_array_internals.shifted_np_points.astype(np.float32)

            np_dst_gaussian_map = cv.warpPerspective(
                np_src_gaussian_map,
                cv.getPerspectiveTransform(
                    np_src_points,
                    np_dst_points,
                    cv.DECOMP_SVD,
                ),
                (bounding_box.width, bounding_box.height),
            )

            # Fill.
            polygon.fill_score_map(
                score_map,
                np_dst_gaussian_map,
                keep_max_value=True,
            )

        return score_map

    def generate_page_char_regression_labels(
        self,
        shape: Tuple[int, int],
        page_char_polygons: Sequence[Polygon],
        rng: RandomGenerator,
    ):
        page_height, page_width = shape

        page_char_regression_labels: List[PageCharRegressionLabel] = []

        for char_idx, polygon in enumerate(page_char_polygons):
            assert len(polygon.points) == 4
            up_left, up_right, down_right, down_left = polygon.points

            # 1. The centroid of char polygon.
            center_point = polygon.get_center_point()
            label = PageCharRegressionLabel(
                char_idx=char_idx,
                tag=PageCharRegressionLabelTag.CENTROID,
                label_point=center_point,
                up_left=up_left,
                up_right=up_right,
                down_right=down_right,
                down_left=down_left,
            )
            # The centroid labeling must be valid.
            assert label.valid
            page_char_regression_labels.append(label)

            # 2. The deviate points.
            bounding_box = polygon.fill_np_array_internals.bounding_box

            # Sample points in shfited bounding box space.
            deviate_points_in_bounding_box = PointList()
            # Some points are invalid, hence double the size of samplings.
            # Also not to sample the points lying on the border to increase the chance of valid.
            for _ in range(2 * self.config.num_deviate_char_regression_labels):
                y = int(rng.integers(1, bounding_box.height - 1))
                x = int(rng.integers(1, bounding_box.width - 1))
                deviate_points_in_bounding_box.append(Point(y=y, x=x))

            # Then transform to the polygon space.
            np_dst_points = polygon.fill_np_array_internals.shifted_np_points.astype(np.float32)
            np_src_points = np.array(
                [
                    (0, 0),
                    (bounding_box.width - 1, 0),
                    (bounding_box.width - 1, bounding_box.height - 1),
                    (0, bounding_box.height - 1),
                ],
                dtype=np.float32,
            )
            trans_mat = cv.getPerspectiveTransform(
                np_src_points,
                np_dst_points,
                cv.DECOMP_SVD,
            )

            deviate_points: List[Point] = []
            for shifted_deviate_point in affine_points(trans_mat, deviate_points_in_bounding_box):
                y = bounding_box.up + shifted_deviate_point.y
                x = bounding_box.left + shifted_deviate_point.x
                assert 0 <= y < page_height
                assert 0 <= x < page_width
                deviate_points.append(Point(y=y, x=x))

            # Build labels.
            num_valid_deviate_char_regression_labels = 0
            for deviate_point in deviate_points:
                if num_valid_deviate_char_regression_labels \
                        >= self.config.num_deviate_char_regression_labels:
                    break
                label = PageCharRegressionLabel(
                    char_idx=char_idx,
                    tag=PageCharRegressionLabelTag.DEVIATE,
                    label_point=deviate_point,
                    up_left=up_left,
                    up_right=up_right,
                    down_right=down_right,
                    down_left=down_left,
                )
                if label.valid:
                    page_char_regression_labels.append(label)
                    num_valid_deviate_char_regression_labels += 1

            if num_valid_deviate_char_regression_labels \
                    < self.config.num_deviate_char_regression_labels:
                logger.warning(f'Cannot sample enough deviate labels for char_polygon={polygon}')

        return page_char_regression_labels

    def run(self, input: PageTextRegionLabelStepInput, rng: RandomGenerator):
        page_text_region_step_output = input.page_text_region_step_output
        page_image = page_text_region_step_output.page_image
        page_char_polygons = page_text_region_step_output.page_char_polygons

        page_score_map = self.generate_page_score_map(
            page_image.shape,
            page_char_polygons,
        )
        page_char_regression_labels = self.generate_page_char_regression_labels(
            page_image.shape,
            page_char_polygons,
            rng,
        )

        return PageTextRegionLabelStepOutput(
            page_score_map=page_score_map,
            page_char_regression_labels=page_char_regression_labels,
        )


page_text_region_label_step_factory = PipelineStepFactory(PageTextRegionLabelStep)
