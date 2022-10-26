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
from typing import cast, Tuple, Sequence, List, Optional
from enum import Enum, unique
import math
import logging

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv
from sklearn.neighbors import KDTree

from vkit.utility import attrs_lazy_field, normalize_to_probs
from vkit.element import Point, PointList, Polygon, Mask, ScoreMap
from vkit.mechanism.distortion.geometric.affine import affine_points
from vkit.engine.char_heatmap import (
    char_heatmap_default_engine_executor_factory,
    CharHeatmapDefaultEngineInitConfig,
)
from ..interface import PipelineStep, PipelineStepFactory
from .page_text_region import PageTextRegionStepOutput

logger = logging.getLogger(__name__)


@attrs.define
class PageTextRegionLabelStepConfig:
    char_heatmap_default_engine_init_config: CharHeatmapDefaultEngineInitConfig = \
        attrs.field(factory=CharHeatmapDefaultEngineInitConfig)

    # 1 centrod + n deviate points.
    num_deviate_char_regression_labels: int = 3
    num_deviate_char_regression_labels_candiates_factor: int = 5


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
    y: float
    x: float

    _distance: Optional[float] = attrs_lazy_field()
    _theta: Optional[float] = attrs_lazy_field()

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

    @classmethod
    def calculate_theta_delta(
        cls,
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
    label_point_y: float
    label_point_x: float
    downsampled_label_point_y: int
    downsampled_label_point_x: int
    up_left: Point
    up_right: Point
    down_right: Point
    down_left: Point

    _up_left_vector: Optional[Vector] = attrs_lazy_field()
    _up_right_vector: Optional[Vector] = attrs_lazy_field()
    _down_right_vector: Optional[Vector] = attrs_lazy_field()
    _down_left_vector: Optional[Vector] = attrs_lazy_field()

    _up_left_to_up_right_angle: Optional[float] = attrs_lazy_field()
    _up_right_to_down_right_angle: Optional[float] = attrs_lazy_field()
    _down_right_to_down_left_angle: Optional[float] = attrs_lazy_field()
    _down_left_to_up_left_angle: Optional[float] = attrs_lazy_field()
    _valid: Optional[bool] = attrs_lazy_field()
    _clockwise_angle_distribution: Optional[Sequence[float]] = attrs_lazy_field()

    def lazy_post_init(self):
        initialized = (self._up_left_vector is not None)
        if initialized:
            return

        self._up_left_vector = Vector(
            y=self.up_left.smooth_y - self.label_point_y,
            x=self.up_left.smooth_x - self.label_point_x,
        )
        self._up_right_vector = Vector(
            y=self.up_right.smooth_y - self.label_point_y,
            x=self.up_right.smooth_x - self.label_point_x,
        )
        self._down_right_vector = Vector(
            y=self.down_right.smooth_y - self.label_point_y,
            x=self.down_right.smooth_x - self.label_point_x,
        )
        self._down_left_vector = Vector(
            y=self.down_left.smooth_y - self.label_point_y,
            x=self.down_left.smooth_x - self.label_point_x,
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

    def to_shifted_page_char_regression_label(self, offset_y: int, offset_x: int):
        assert self.valid
        # Only can be called before downsampling.
        assert self.label_point_y == self.downsampled_label_point_y
        assert self.label_point_x == self.downsampled_label_point_x

        label_point_y = cast(int, self.label_point_y + offset_y)
        label_point_x = cast(int, self.label_point_x + offset_x)
        up_left = self.up_left.to_shifted_point(offset_y=offset_y, offset_x=offset_x)
        up_right = self.up_right.to_shifted_point(offset_y=offset_y, offset_x=offset_x)
        down_right = self.down_right.to_shifted_point(offset_y=offset_y, offset_x=offset_x)
        down_left = self.down_left.to_shifted_point(offset_y=offset_y, offset_x=offset_x)

        shifted = PageCharRegressionLabel(
            char_idx=self.char_idx,
            tag=self.tag,
            label_point_y=label_point_y,
            label_point_x=label_point_x,
            downsampled_label_point_y=label_point_y,
            downsampled_label_point_x=label_point_x,
            up_left=up_left,
            up_right=up_right,
            down_right=down_right,
            down_left=down_left,
        )

        # Avoid recalculate the labelings.
        shifted._up_left_vector = self._up_left_vector
        shifted._up_right_vector = self._up_right_vector
        shifted._down_right_vector = self._down_right_vector
        shifted._down_left_vector = self._down_left_vector
        shifted._up_left_to_up_right_angle = self._up_left_to_up_right_angle
        shifted._up_right_to_down_right_angle = self._up_right_to_down_right_angle
        shifted._down_right_to_down_left_angle = self._down_right_to_down_left_angle
        shifted._down_left_to_up_left_angle = self._down_left_to_up_left_angle
        shifted._valid = self._valid
        shifted._clockwise_angle_distribution = self._clockwise_angle_distribution

        return shifted

    def to_downsampled_page_char_regression_label(self, downsample_labeling_factor: int):
        assert self.valid
        # Only can be called before downsampling.
        assert self.label_point_y == self.downsampled_label_point_y
        assert self.label_point_x == self.downsampled_label_point_x

        downsampled_label_point_y = int(self.label_point_y // downsample_labeling_factor)
        downsampled_label_point_x = int(self.label_point_x // downsample_labeling_factor)

        # label_point_* are shifted to the center of upsampled positions.
        offset = (downsample_labeling_factor - 1) / 2
        label_point_y = downsampled_label_point_y * downsample_labeling_factor + offset
        label_point_x = downsampled_label_point_x * downsample_labeling_factor + offset

        downsampled_page_char_regression_label = PageCharRegressionLabel(
            char_idx=self.char_idx,
            tag=self.tag,
            label_point_y=label_point_y,
            label_point_x=label_point_x,
            downsampled_label_point_y=downsampled_label_point_y,
            downsampled_label_point_x=downsampled_label_point_x,
            up_left=self.up_left,
            up_right=self.up_right,
            down_right=self.down_right,
            down_left=self.down_left,
        )
        return downsampled_page_char_regression_label

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
    page_char_mask: Mask
    page_char_height_score_map: ScoreMap
    page_char_gaussian_score_map: ScoreMap
    page_char_regression_labels: Sequence[PageCharRegressionLabel]


class PageTextRegionLabelStep(
    PipelineStep[
        PageTextRegionLabelStepConfig,
        PageTextRegionLabelStepInput,
        PageTextRegionLabelStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageTextRegionLabelStepConfig):
        super().__init__(config)

        self.char_heatmap_default_engine_executor = \
            char_heatmap_default_engine_executor_factory.create(
                self.config.char_heatmap_default_engine_init_config
            )

    @classmethod
    def generate_page_char_mask(
        cls,
        shape: Tuple[int, int],
        page_char_polygons: Sequence[Polygon],
    ):
        page_char_mask = Mask.from_shape(shape)
        for polygon in page_char_polygons:
            polygon.fill_mask(page_char_mask)
        return page_char_mask

    @classmethod
    def generate_page_char_height_score_map(
        cls,
        shape: Tuple[int, int],
        page_char_polygons: Sequence[Polygon],
    ):
        page_char_height_score_map = ScoreMap.from_shape(shape, is_prob=False)
        for polygon in page_char_polygons:
            polygon.fill_score_map(
                page_char_height_score_map,
                value=polygon.get_rectangular_height(),
            )
        return page_char_height_score_map

    def generate_page_char_gaussian_score_map(
        self,
        shape: Tuple[int, int],
        page_char_polygons: Sequence[Polygon],
        rng: RandomGenerator,
    ):
        height, width = shape
        char_heatmap = self.char_heatmap_default_engine_executor.run(
            {
                'height': height,
                'width': width,
                'char_polygons': page_char_polygons,
            },
            rng,
        )
        return char_heatmap.score_map

    def generate_page_char_regression_labels(
        self,
        shape: Tuple[int, int],
        page_char_polygons: Sequence[Polygon],
        rng: RandomGenerator,
    ):
        page_height, page_width = shape

        # Build a KD tree to for removing deviate point that is too close to another center point.
        center_points = PointList()
        for polygon in page_char_polygons:
            center_points.append(polygon.get_center_point())
        kd_tree = KDTree(center_points.to_np_array())

        page_char_regression_labels: List[PageCharRegressionLabel] = []

        for char_idx, (polygon, center_point) in enumerate(zip(page_char_polygons, center_points)):
            assert polygon.num_points == 4
            up_left, up_right, down_right, down_left = polygon.points

            # 1. The centroid of char polygon.
            label = PageCharRegressionLabel(
                char_idx=char_idx,
                tag=PageCharRegressionLabelTag.CENTROID,
                label_point_y=center_point.y,
                label_point_x=center_point.x,
                downsampled_label_point_y=center_point.y,
                downsampled_label_point_x=center_point.x,
                up_left=up_left,
                up_right=up_right,
                down_right=down_right,
                down_left=down_left,
            )
            # The centroid labeling must be valid.
            assert label.valid
            page_char_regression_labels.append(label)

            # 2. The deviate points.
            bounding_box = polygon.bounding_box

            # Sample points in shfited bounding box space.
            deviate_points_in_bounding_box = PointList()
            # Some points are invalid, hence multiply the number of samplings by a factor.
            # Also not to sample the points lying on the border to increase the chance of valid.
            for _ in range(
                self.config.num_deviate_char_regression_labels_candiates_factor
                * self.config.num_deviate_char_regression_labels
            ):
                y = int(rng.integers(1, bounding_box.height - 1))
                x = int(rng.integers(1, bounding_box.width - 1))
                deviate_points_in_bounding_box.append(Point.create(y=y, x=x))

            # Then transform to the polygon space.
            np_src_points = np.asarray(
                [
                    (0, 0),
                    (bounding_box.width - 1, 0),
                    (bounding_box.width - 1, bounding_box.height - 1),
                    (0, bounding_box.height - 1),
                ],
                dtype=np.float32,
            )
            np_dst_points = polygon.internals.np_self_relative_points
            trans_mat = cv.getPerspectiveTransform(
                np_src_points,
                np_dst_points,
                cv.DECOMP_SVD,
            )

            deviate_points = PointList()
            for shifted_deviate_point in affine_points(
                trans_mat,
                deviate_points_in_bounding_box.to_point_tuple(),
            ):
                y = bounding_box.up + shifted_deviate_point.y
                x = bounding_box.left + shifted_deviate_point.x
                assert 0 <= y < page_height
                assert 0 <= x < page_width
                deviate_points.append(Point.create(y=y, x=x))

            # Remove those are too close to another center point.
            _, np_kd_nbr_indices = kd_tree.query(deviate_points.to_np_array())
            preserve_flags: List[bool] = [
                idx == char_idx for idx in np_kd_nbr_indices[:, 0].tolist()
            ]

            # Build labels.
            num_valid_deviate_char_regression_labels = 0
            for deviate_point, preserve_flag in zip(deviate_points, preserve_flags):
                if num_valid_deviate_char_regression_labels \
                        >= self.config.num_deviate_char_regression_labels:
                    break

                if not preserve_flag:
                    continue

                label = PageCharRegressionLabel(
                    char_idx=char_idx,
                    tag=PageCharRegressionLabelTag.DEVIATE,
                    label_point_y=deviate_point.y,
                    label_point_x=deviate_point.x,
                    downsampled_label_point_y=deviate_point.y,
                    downsampled_label_point_x=deviate_point.x,
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

        page_char_mask = self.generate_page_char_mask(
            page_image.shape,
            page_char_polygons,
        )
        page_char_height_score_map = self.generate_page_char_height_score_map(
            page_image.shape,
            page_char_polygons,
        )

        page_char_gaussian_score_map = self.generate_page_char_gaussian_score_map(
            page_image.shape,
            page_char_polygons,
            rng,
        )
        page_char_regression_labels = self.generate_page_char_regression_labels(
            page_image.shape,
            page_char_polygons,
            rng,
        )

        return PageTextRegionLabelStepOutput(
            page_char_mask=page_char_mask,
            page_char_height_score_map=page_char_height_score_map,
            page_char_gaussian_score_map=page_char_gaussian_score_map,
            page_char_regression_labels=page_char_regression_labels,
        )


page_text_region_label_step_factory = PipelineStepFactory(PageTextRegionLabelStep)
