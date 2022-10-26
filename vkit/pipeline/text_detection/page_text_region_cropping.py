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
from typing import Sequence, Mapping, Tuple, DefaultDict, List, Optional
from collections import defaultdict
import itertools
import warnings

import attrs
from numpy.random import Generator as RandomGenerator
from shapely.errors import ShapelyDeprecationWarning
from shapely.strtree import STRtree
from shapely.geometry import Point as ShapelyPoint
import cv2 as cv

from vkit.element import Box, Mask, ScoreMap, Image
from vkit.mechanism.distortion import rotate
from vkit.mechanism.cropper import Cropper
from ..interface import PipelineStep, PipelineStepFactory
from .page_cropping import PageCroppingStepOutput
from .page_text_region import PageTextRegionStepOutput
from .page_text_region_label import (
    PageCharRegressionLabelTag,
    PageCharRegressionLabel,
    PageTextRegionLabelStepOutput,
)

# Shapely version has been explicitly locked under 2.0, hence ignore this warning.
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)


@attrs.define
class PageTextRegionCroppingStepConfig:
    core_size: int
    pad_size: int
    num_samples_factor_relative_to_num_cropped_pages: float = 1.0
    num_centroid_points_min: int = 10
    num_deviate_points_min: int = 10
    pad_value: int = 0
    enable_downsample_labeling: bool = True
    downsample_labeling_factor: int = 2


@attrs.define
class PageTextRegionCroppingStepInput:
    page_cropping_step_output: PageCroppingStepOutput
    page_text_region_step_output: PageTextRegionStepOutput
    page_text_region_label_step_output: PageTextRegionLabelStepOutput


@attrs.define
class DownsampledLabel:
    shape: Tuple[int, int]
    page_char_mask: Mask
    page_char_height_score_map: ScoreMap
    page_char_gaussian_score_map: ScoreMap
    page_char_regression_labels: Sequence[PageCharRegressionLabel]
    core_box: Box


@attrs.define
class CroppedPageTextRegion:
    page_image: Image
    page_char_mask: Mask
    page_char_height_score_map: ScoreMap
    page_char_gaussian_score_map: ScoreMap
    page_char_regression_labels: Sequence[PageCharRegressionLabel]
    core_box: Box
    downsampled_label: Optional[DownsampledLabel]


@attrs.define
class PageTextRegionCroppingStepOutput:
    cropped_page_text_regions: Sequence[CroppedPageTextRegion]


class PageTextRegionCroppingStep(
    PipelineStep[
        PageTextRegionCroppingStepConfig,
        PageTextRegionCroppingStepInput,
        PageTextRegionCroppingStepOutput,
    ]
):  # yapf: disable

    @classmethod
    def build_strtree_for_page_char_regression_labels(
        cls,
        labels: Sequence[PageCharRegressionLabel],
    ):
        shapely_points: List[ShapelyPoint] = []

        xy_pair_to_labels: DefaultDict[
            Tuple[int, int],
            List[PageCharRegressionLabel],
        ] = defaultdict(list)  # yapf: disable

        for label in labels:
            assert isinstance(label.label_point_x, int)
            assert isinstance(label.label_point_y, int)
            xy_pair = (label.label_point_x, label.label_point_y)
            shapely_points.append(ShapelyPoint(*xy_pair))
            xy_pair_to_labels[xy_pair].append(label)

        strtree = STRtree(shapely_points)
        return strtree, xy_pair_to_labels

    def sample_cropped_page_text_regions(
        self,
        page_image: Image,
        shape_before_rotate: Tuple[int, int],
        rotate_angle: int,
        page_char_mask: Mask,
        page_char_height_score_map: ScoreMap,
        page_char_gaussian_score_map: ScoreMap,
        centroid_strtree: STRtree,
        centroid_xy_pair_to_labels: Mapping[Tuple[int, int], Sequence[PageCharRegressionLabel]],
        deviate_strtree: STRtree,
        deviate_xy_pair_to_labels: Mapping[Tuple[int, int], Sequence[PageCharRegressionLabel]],
        rng: RandomGenerator,
    ):
        if rotate_angle != 0:
            cropper_before_rotate = Cropper.create(
                shape=shape_before_rotate,
                core_size=self.config.core_size,
                pad_size=self.config.pad_size,
                pad_value=self.config.pad_value,
                rng=rng,
            )
            origin_box_before_rotate = cropper_before_rotate.cropper_state.origin_box
            center_point_before_rotate = origin_box_before_rotate.get_center_point()

            rotated_result = rotate.distort(
                {'angle': rotate_angle},
                shapable_or_shape=shape_before_rotate,
                point=center_point_before_rotate,
            )
            assert rotated_result.shape == page_image.shape
            center_point = rotated_result.point
            assert center_point

            cropper = Cropper.create_from_center_point(
                shape=page_image.shape,
                core_size=self.config.core_size,
                pad_size=self.config.pad_size,
                pad_value=self.config.pad_value,
                center_point=center_point,
            )

        else:
            cropper = Cropper.create(
                shape=page_image.shape,
                core_size=self.config.core_size,
                pad_size=self.config.pad_size,
                pad_value=self.config.pad_value,
                rng=rng,
            )

        # Pick labels.
        origin_core_shapely_polygon = cropper.origin_core_box.to_shapely_polygon()

        centroid_labels: List[PageCharRegressionLabel] = []
        for shapely_point in centroid_strtree.query(origin_core_shapely_polygon):
            if not origin_core_shapely_polygon.intersects(shapely_point):
                continue
            assert isinstance(shapely_point, ShapelyPoint)
            centroid_xy_pair = (int(shapely_point.x), int(shapely_point.y))
            centroid_labels.extend(centroid_xy_pair_to_labels[centroid_xy_pair])

        deviate_labels: List[PageCharRegressionLabel] = []
        for shapely_point in deviate_strtree.query(origin_core_shapely_polygon):
            if not origin_core_shapely_polygon.intersects(shapely_point):
                continue
            assert isinstance(shapely_point, ShapelyPoint)
            deviate_xy_pair = (int(shapely_point.x), int(shapely_point.y))
            deviate_labels.extend(deviate_xy_pair_to_labels[deviate_xy_pair])

        if len(centroid_labels) < self.config.num_centroid_points_min \
                or len(deviate_labels) < self.config.num_deviate_points_min:
            return None

        # Shift labels.
        offset_y = cropper.target_box.up - cropper.origin_box.up
        offset_x = cropper.target_box.left - cropper.origin_box.left
        shifted_centroid_labels = [
            centroid_label.to_shifted_page_char_regression_label(
                offset_y=offset_y,
                offset_x=offset_x,
            ) for centroid_label in centroid_labels
        ]
        shifted_deviate_labels = [
            deviate_label.to_shifted_page_char_regression_label(
                offset_y=offset_y,
                offset_x=offset_x,
            ) for deviate_label in deviate_labels
        ]

        # Crop image and score map.
        page_image = cropper.crop_image(page_image)
        page_char_mask = cropper.crop_mask(
            page_char_mask,
            core_only=True,
        )
        page_char_height_score_map = cropper.crop_score_map(
            page_char_height_score_map,
            core_only=True,
        )
        page_char_gaussian_score_map = cropper.crop_score_map(
            page_char_gaussian_score_map,
            core_only=True,
        )

        downsampled_label: Optional[DownsampledLabel] = None
        if self.config.enable_downsample_labeling:
            downsample_labeling_factor = self.config.downsample_labeling_factor

            assert cropper.crop_size % downsample_labeling_factor == 0
            downsampled_size = cropper.crop_size // downsample_labeling_factor
            downsampled_shape = (downsampled_size, downsampled_size)

            assert self.config.pad_size % downsample_labeling_factor == 0
            assert self.config.core_size % downsample_labeling_factor == 0
            assert cropper.core_box.height == cropper.core_box.width == self.config.core_size

            downsampled_pad_size = self.config.pad_size // downsample_labeling_factor
            downsampled_core_size = self.config.core_size // downsample_labeling_factor

            downsampled_core_begin = downsampled_pad_size
            downsampled_core_end = downsampled_core_begin + downsampled_core_size - 1
            downsampled_core_box = Box(
                up=downsampled_core_begin,
                down=downsampled_core_end,
                left=downsampled_core_begin,
                right=downsampled_core_end,
            )

            downsampled_page_char_mask = page_char_mask.to_box_detached()
            downsampled_page_char_mask = \
                downsampled_page_char_mask.to_resized_mask(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_page_char_height_score_map = page_char_height_score_map.to_box_detached()
            downsampled_page_char_height_score_map = \
                downsampled_page_char_height_score_map.to_resized_score_map(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_page_char_gaussian_score_map = \
                page_char_gaussian_score_map.to_box_detached()
            downsampled_page_char_gaussian_score_map = \
                downsampled_page_char_gaussian_score_map.to_resized_score_map(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_page_char_regression_labels = [
                label.to_downsampled_page_char_regression_label(
                    self.config.downsample_labeling_factor
                ) for label in itertools.chain(shifted_centroid_labels, shifted_deviate_labels)
            ]

            downsampled_label = DownsampledLabel(
                shape=downsampled_shape,
                page_char_mask=downsampled_page_char_mask,
                page_char_height_score_map=downsampled_page_char_height_score_map,
                page_char_gaussian_score_map=downsampled_page_char_gaussian_score_map,
                page_char_regression_labels=downsampled_page_char_regression_labels,
                core_box=downsampled_core_box,
            )

        return CroppedPageTextRegion(
            page_image=page_image,
            page_char_mask=page_char_mask,
            page_char_height_score_map=page_char_height_score_map,
            page_char_gaussian_score_map=page_char_gaussian_score_map,
            page_char_regression_labels=shifted_centroid_labels + shifted_deviate_labels,
            core_box=cropper.core_box,
            downsampled_label=downsampled_label,
        )

    def run(self, input: PageTextRegionCroppingStepInput, rng: RandomGenerator):
        page_cropping_step_output = input.page_cropping_step_output
        num_cropped_pages = len(page_cropping_step_output.cropped_pages)

        page_text_region_step_output = input.page_text_region_step_output
        page_image = page_text_region_step_output.page_image
        shape_before_rotate = page_text_region_step_output.shape_before_rotate
        rotate_angle = page_text_region_step_output.rotate_angle

        page_text_region_label_step_output = input.page_text_region_label_step_output
        page_char_mask = page_text_region_label_step_output.page_char_mask
        page_char_height_score_map = page_text_region_label_step_output.page_char_height_score_map
        page_char_gaussian_score_map = \
            page_text_region_label_step_output.page_char_gaussian_score_map
        page_char_regression_labels = \
            page_text_region_label_step_output.page_char_regression_labels

        (
            centroid_strtree,
            centroid_xy_pair_to_labels,
        ) = self.build_strtree_for_page_char_regression_labels([
            label for label in page_char_regression_labels
            if label.tag == PageCharRegressionLabelTag.CENTROID
        ])
        (
            deviate_strtree,
            deviate_xy_pair_to_labels,
        ) = self.build_strtree_for_page_char_regression_labels([
            label for label in page_char_regression_labels
            if label.tag == PageCharRegressionLabelTag.DEVIATE
        ])

        num_samples = round(
            self.config.num_samples_factor_relative_to_num_cropped_pages * num_cropped_pages
        )

        run_count_max = max(3, 2 * num_samples)
        run_count = 0

        cropped_page_text_regions: List[CroppedPageTextRegion] = []

        while len(cropped_page_text_regions) < num_samples and run_count < run_count_max:
            cropped_page_text_region = self.sample_cropped_page_text_regions(
                page_image=page_image,
                shape_before_rotate=shape_before_rotate,
                rotate_angle=rotate_angle,
                page_char_mask=page_char_mask,
                page_char_height_score_map=page_char_height_score_map,
                page_char_gaussian_score_map=page_char_gaussian_score_map,
                centroid_strtree=centroid_strtree,
                centroid_xy_pair_to_labels=centroid_xy_pair_to_labels,
                deviate_strtree=deviate_strtree,
                deviate_xy_pair_to_labels=deviate_xy_pair_to_labels,
                rng=rng,
            )
            if cropped_page_text_region:
                cropped_page_text_regions.append(cropped_page_text_region)
            run_count += 1

        return PageTextRegionCroppingStepOutput(
            cropped_page_text_regions=cropped_page_text_regions,
        )


page_text_region_cropping_step_factory = PipelineStepFactory(PageTextRegionCroppingStep)
