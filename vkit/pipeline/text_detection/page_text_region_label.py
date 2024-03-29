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
from typing import Tuple, Sequence, List, Optional, Mapping, Any
from enum import Enum, unique
import math
import logging

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv
from sklearn.neighbors import KDTree

from vkit.utility import attrs_lazy_field, unwrap_optional_field, normalize_to_probs
from vkit.element import Point, PointList, Box, Polygon, Mask, ScoreMap
from vkit.mechanism.distortion.geometric.affine import affine_points
from vkit.engine.char_heatmap import (
    char_heatmap_default_engine_executor_factory,
    CharHeatmapDefaultEngineInitConfig,
)
from vkit.engine.char_mask import (
    char_mask_engine_executor_aggregator_factory,
    CharMaskEngineRunConfig,
)
from ..interface import PipelineStep, PipelineStepFactory
from .page_text_region import PageTextRegionStepOutput

logger = logging.getLogger(__name__)


@attrs.define
class PageTextRegionLabelStepConfig:
    char_heatmap_default_engine_init_config: CharHeatmapDefaultEngineInitConfig = \
        attrs.field(factory=CharHeatmapDefaultEngineInitConfig)
    char_mask_engine_config: Mapping[str, Any] = attrs.field(factory=lambda: {'type': 'default'})

    # 1 centrod + n deviate points.
    num_deviate_char_regression_labels: int = 1
    num_deviate_char_regression_labels_candiates_factor: int = 3


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
        return unwrap_optional_field(self._distance)

    @property
    def theta(self):
        self.lazy_post_init()
        return unwrap_optional_field(self._theta)

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
    label_point_smooth_y: float
    label_point_smooth_x: float
    downsampled_label_point_y: int
    downsampled_label_point_x: int
    up_left: Point
    up_right: Point
    down_right: Point
    down_left: Point

    is_downsampled: bool = False
    downsample_labeling_factor: int = 1

    _bounding_smooth_up: Optional[float] = attrs_lazy_field()
    _bounding_smooth_down: Optional[float] = attrs_lazy_field()
    _bounding_smooth_left: Optional[float] = attrs_lazy_field()
    _bounding_smooth_right: Optional[float] = attrs_lazy_field()
    _bounding_orientation_idx: Optional[int] = attrs_lazy_field()

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

    @property
    def corner_points(self):
        yield from (self.up_left, self.up_right, self.down_right, self.down_left)

    @classmethod
    def get_bounding_orientation_idx(cls, down_left: Point, down_right: Point):
        vector = Vector(
            y=down_right.smooth_y - down_left.smooth_y,
            x=down_right.smooth_x - down_left.smooth_x,
        )
        #        0
        #  ┌───────────┐
        #  │           │
        # 2│           │3
        #  │           │
        #  └───────────┘
        #        1
        factor = vector.theta / PI
        if 1.75 <= factor or factor < 0.25:
            return 1
        elif 0.25 <= factor < 0.75:
            return 2
        elif 0.75 <= factor < 1.25:
            return 0
        elif 1.25 <= factor:
            return 3
        else:
            raise RuntimeError()

    def lazy_post_init(self):
        if self._bounding_smooth_up is None:
            self._bounding_smooth_up = min(point.smooth_y for point in self.corner_points)
            self._bounding_smooth_down = max(point.smooth_y for point in self.corner_points)
            self._bounding_smooth_left = min(point.smooth_x for point in self.corner_points)
            self._bounding_smooth_right = max(point.smooth_x for point in self.corner_points)
            self._bounding_orientation_idx = self.get_bounding_orientation_idx(
                down_left=self.down_left,
                down_right=self.down_right,
            )

        initialized = (self._up_left_vector is not None)
        if initialized:
            return

        self._up_left_vector = Vector(
            y=self.up_left.smooth_y - self.label_point_smooth_y,
            x=self.up_left.smooth_x - self.label_point_smooth_x,
        )
        self._up_right_vector = Vector(
            y=self.up_right.smooth_y - self.label_point_smooth_y,
            x=self.up_right.smooth_x - self.label_point_smooth_x,
        )
        self._down_right_vector = Vector(
            y=self.down_right.smooth_y - self.label_point_smooth_y,
            x=self.down_right.smooth_x - self.label_point_smooth_x,
        )
        self._down_left_vector = Vector(
            y=self.down_left.smooth_y - self.label_point_smooth_y,
            x=self.down_left.smooth_x - self.label_point_smooth_x,
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

    def copy(self, with_non_bounding_related_lazy_fields: bool = False):
        copied = attrs.evolve(self)

        if with_non_bounding_related_lazy_fields:
            # NOTE: Bounding box related properties are not copied.
            copied._up_left_vector = self._up_left_vector
            copied._up_right_vector = self._up_right_vector
            copied._down_right_vector = self._down_right_vector
            copied._down_left_vector = self._down_left_vector
            copied._up_left_to_up_right_angle = self._up_left_to_up_right_angle
            copied._up_right_to_down_right_angle = self._up_right_to_down_right_angle
            copied._down_right_to_down_left_angle = self._down_right_to_down_left_angle
            copied._down_left_to_up_left_angle = self._down_left_to_up_left_angle
            copied._valid = self._valid
            copied._clockwise_angle_distribution = self._clockwise_angle_distribution

        return copied

    def to_shifted_page_char_regression_label(self, offset_y: int, offset_x: int):
        assert self.valid and not self.is_downsampled

        # Shift operation doesn't change the lazy fields.
        shifted = self.copy(with_non_bounding_related_lazy_fields=True)

        shifted.label_point_smooth_y = self.label_point_smooth_y + offset_y
        shifted.label_point_smooth_x = self.label_point_smooth_x + offset_x
        shifted.downsampled_label_point_y = int(shifted.label_point_smooth_y)
        shifted.downsampled_label_point_x = int(shifted.label_point_smooth_x)
        shifted.up_left = self.up_left.to_shifted_point(offset_y=offset_y, offset_x=offset_x)
        shifted.up_right = self.up_right.to_shifted_point(offset_y=offset_y, offset_x=offset_x)
        shifted.down_right = self.down_right.to_shifted_point(offset_y=offset_y, offset_x=offset_x)
        shifted.down_left = self.down_left.to_shifted_point(offset_y=offset_y, offset_x=offset_x)

        return shifted

    def to_downsampled_page_char_regression_label(self, downsample_labeling_factor: int):
        assert self.valid and not self.is_downsampled

        # Downsample operation doesn't change the lazy fields.
        downsampled = self.copy(with_non_bounding_related_lazy_fields=True)
        # Mark as downsampled hence disables shift & downsample opts.
        downsampled.is_downsampled = True
        # Should be helpful in training.
        downsampled.downsample_labeling_factor = downsample_labeling_factor

        downsampled.downsampled_label_point_y = \
            int(self.label_point_smooth_y // downsample_labeling_factor)
        downsampled.downsampled_label_point_x = \
            int(self.label_point_smooth_x // downsample_labeling_factor)

        return downsampled

    @property
    def bounding_smooth_up(self):
        self.lazy_post_init()
        return unwrap_optional_field(self._bounding_smooth_up)

    @property
    def bounding_smooth_down(self):
        self.lazy_post_init()
        return unwrap_optional_field(self._bounding_smooth_down)

    @property
    def bounding_smooth_left(self):
        self.lazy_post_init()
        return unwrap_optional_field(self._bounding_smooth_left)

    @property
    def bounding_smooth_right(self):
        self.lazy_post_init()
        return unwrap_optional_field(self._bounding_smooth_right)

    @property
    def bounding_center_point(self):
        return Point.create(
            y=(self.bounding_smooth_up + self.bounding_smooth_down) / 2,
            x=(self.bounding_smooth_left + self.bounding_smooth_right) / 2,
        )

    @property
    def bounding_smooth_shape(self):
        height = self.bounding_smooth_down - self.bounding_smooth_up
        width = self.bounding_smooth_right - self.bounding_smooth_left
        return height, width

    @property
    def bounding_orientation_idx(self):
        self.lazy_post_init()
        return unwrap_optional_field(self._bounding_orientation_idx)

    @property
    def valid(self):
        self.lazy_post_init()
        return unwrap_optional_field(self._valid)

    def generate_up_left_offsets(self):
        self.lazy_post_init()
        up_left_vector = unwrap_optional_field(self._up_left_vector)
        return up_left_vector.y, up_left_vector.x

    def generate_clockwise_angle_distribution(self):
        self.lazy_post_init()
        return unwrap_optional_field(self._clockwise_angle_distribution)

    def generate_clockwise_distances(self):
        self.lazy_post_init()
        return (
            unwrap_optional_field(self._up_left_vector).distance,
            unwrap_optional_field(self._up_right_vector).distance,
            unwrap_optional_field(self._down_right_vector).distance,
            unwrap_optional_field(self._down_left_vector).distance,
        )


@attrs.define
class PageTextRegionLabelStepOutput:
    page_char_mask: Mask
    page_char_height_score_map: ScoreMap
    page_char_gaussian_score_map: ScoreMap
    page_char_regression_labels: Sequence[PageCharRegressionLabel]
    page_char_bounding_box_mask: Mask


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
        self.char_mask_engine_executor = \
            char_mask_engine_executor_aggregator_factory.create_engine_executor(
                self.config.char_mask_engine_config
            )

    def generate_page_char_mask(
        self,
        shape: Tuple[int, int],
        page_inactive_mask: Mask,
        page_char_polygons: Sequence[Polygon],
        page_text_region_polygons: Sequence[Polygon],
        page_char_polygon_text_region_polygon_indices: Sequence[int],
    ):
        height, width = shape
        result = self.char_mask_engine_executor.run(
            CharMaskEngineRunConfig(
                height=height,
                width=width,
                char_polygons=page_char_polygons,
                char_bounding_polygons=[
                    page_text_region_polygons[idx]
                    for idx in page_char_polygon_text_region_polygon_indices
                ],
            ),
        )

        page_inactive_mask.fill_mask(result.combined_chars_mask, 0)

        return result.combined_chars_mask, result.char_masks

    @classmethod
    def generate_page_char_height_score_map(
        cls,
        shape: Tuple[int, int],
        page_inactive_mask: Mask,
        page_char_polygons: Sequence[Polygon],
        fill_score_map_char_masks: Optional[Sequence[Mask]],
    ):
        rectangular_heights = [
            char_polygon.get_rectangular_height() for char_polygon in page_char_polygons
        ]
        sorted_indices: Tuple[int, ...] = tuple(reversed(np.asarray(rectangular_heights).argsort()))

        page_char_height_score_map = ScoreMap.from_shape(shape, is_prob=False)
        for idx in sorted_indices:
            char_polygon = page_char_polygons[idx]
            rectangular_height = rectangular_heights[idx]
            if fill_score_map_char_masks is None:
                char_polygon.fill_score_map(
                    page_char_height_score_map,
                    value=rectangular_height,
                )
            else:
                char_mask = fill_score_map_char_masks[idx]
                char_mask.fill_score_map(
                    page_char_height_score_map,
                    value=rectangular_height,
                )

        page_inactive_mask.fill_score_map(page_char_height_score_map, 0.0)

        return page_char_height_score_map

    def generate_page_char_gaussian_score_map(
        self,
        shape: Tuple[int, int],
        page_char_polygons: Sequence[Polygon],
    ):
        height, width = shape
        char_heatmap = self.char_heatmap_default_engine_executor.run({
            'height': height,
            'width': width,
            'char_polygons': page_char_polygons,
        })
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
                label_point_smooth_y=center_point.smooth_y,
                label_point_smooth_x=center_point.smooth_x,
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
            if self.config.num_deviate_char_regression_labels <= 0:
                # Generating deviate points are optional.
                continue

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
                y = bounding_box.up + shifted_deviate_point.smooth_y
                x = bounding_box.left + shifted_deviate_point.smooth_x
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
                    label_point_smooth_y=deviate_point.smooth_y,
                    label_point_smooth_x=deviate_point.smooth_x,
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

    def generate_page_char_bounding_box_mask(
        self,
        shape: Tuple[int, int],
        page_char_regression_labels: Sequence[PageCharRegressionLabel],
    ):
        page_char_bounding_box_mask = Mask.from_shape(shape)
        for page_char_regression_label in page_char_regression_labels:
            box = Box(
                up=math.floor(page_char_regression_label.bounding_smooth_up),
                down=math.ceil(page_char_regression_label.bounding_smooth_down),
                left=math.floor(page_char_regression_label.bounding_smooth_left),
                right=math.ceil(page_char_regression_label.bounding_smooth_right),
            )
            box.fill_mask(page_char_bounding_box_mask)
        return page_char_bounding_box_mask

    def run(self, input: PageTextRegionLabelStepInput, rng: RandomGenerator):
        page_text_region_step_output = input.page_text_region_step_output
        page_image = page_text_region_step_output.page_image
        page_active_mask = page_text_region_step_output.page_active_mask
        page_char_polygons = page_text_region_step_output.page_char_polygons
        page_text_region_polygons = page_text_region_step_output.page_text_region_polygons
        page_char_polygon_text_region_polygon_indices = \
            page_text_region_step_output.page_char_polygon_text_region_polygon_indices

        page_inactive_mask = page_active_mask.to_inverted_mask()
        page_char_mask, fill_score_map_char_masks = self.generate_page_char_mask(
            shape=page_image.shape,
            page_inactive_mask=page_inactive_mask,
            page_char_polygons=page_char_polygons,
            page_text_region_polygons=page_text_region_polygons,
            page_char_polygon_text_region_polygon_indices=(
                page_char_polygon_text_region_polygon_indices
            ),
        )

        # NOTE: page_char_height_score_map is different from the one defined in page distortion.
        # TODO: Resolve the inconsistency.
        page_char_height_score_map = self.generate_page_char_height_score_map(
            shape=page_image.shape,
            page_inactive_mask=page_inactive_mask,
            page_char_polygons=page_char_polygons,
            fill_score_map_char_masks=fill_score_map_char_masks,
        )

        page_char_gaussian_score_map = self.generate_page_char_gaussian_score_map(
            page_image.shape,
            page_char_polygons,
        )

        page_char_regression_labels = self.generate_page_char_regression_labels(
            page_image.shape,
            page_char_polygons,
            rng,
        )

        page_char_bounding_box_mask = self.generate_page_char_bounding_box_mask(
            page_image.shape,
            page_char_regression_labels,
        )

        return PageTextRegionLabelStepOutput(
            page_char_mask=page_char_mask,
            page_char_height_score_map=page_char_height_score_map,
            page_char_gaussian_score_map=page_char_gaussian_score_map,
            page_char_regression_labels=page_char_regression_labels,
            page_char_bounding_box_mask=page_char_bounding_box_mask,
        )


page_text_region_label_step_factory = PipelineStepFactory(PageTextRegionLabelStep)
