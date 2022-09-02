from typing import Sequence, Mapping, Tuple, DefaultDict, List
from collections import defaultdict

import attrs
from numpy.random import Generator as RandomGenerator
from shapely.strtree import STRtree
from shapely.geometry import Point as ShapelyPoint

from vkit.element import Box, ScoreMap, Image, Cropper
from vkit.engine.distortion.geometric.affine import rotate
from ..interface import PipelineStep, PipelineStepFactory
from .page_cropping import PageCroppingStepOutput
from .page_text_region import PageTextRegionStepOutput
from .page_text_region_label import (
    PageCharRegressionLabelTag,
    PageCharRegressionLabel,
    PageTextRegionLabelStepOutput,
)


@attrs.define
class PageTextRegionCroppingStepConfig:
    core_size: int
    pad_size: int
    num_samples_factor_relative_to_num_cropped_pages: float = 1.0
    num_centroid_points_min: int = 10
    num_deviate_points_min: int = 10
    pad_value: int = 0


@attrs.define
class PageTextRegionCroppingStepInput:
    page_cropping_step_output: PageCroppingStepOutput
    page_text_region_step_output: PageTextRegionStepOutput
    page_text_region_label_step_output: PageTextRegionLabelStepOutput


@attrs.define
class CroppedPageTextRegion:
    page_image: Image
    page_score_map: ScoreMap
    page_char_regression_labels: Sequence[PageCharRegressionLabel]
    core_box: Box


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

    @staticmethod
    def build_strtree_for_page_char_regression_labels(labels: Sequence[PageCharRegressionLabel]):
        shapely_points: List[ShapelyPoint] = []

        xy_pair_to_labels: DefaultDict[
            Tuple[int, int],
            List[PageCharRegressionLabel],
        ] = defaultdict(list)  # yapf: disable

        for label in labels:
            label_point = label.label_point
            shapely_points.append(ShapelyPoint(label_point.x, label_point.y))
            xy_pair_to_labels[label_point.to_xy_pair()].append(label)

        strtree = STRtree(shapely_points)
        return strtree, xy_pair_to_labels

    def sample_cropped_page_text_regions(
        self,
        page_image: Image,
        shape_before_rotate: Tuple[int, int],
        rotate_angle: int,
        page_score_map: ScoreMap,
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
        origin_core_shapely_polygon = cropper.origin_core_box.to_polygon().to_shapely_polygon()

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
        y_offset = cropper.target_box.up - cropper.origin_box.up
        x_offset = cropper.target_box.left - cropper.origin_box.left
        shifted_centroid_labels = [
            centroid_label.to_shifted_page_char_regression_label(
                y_offset=y_offset,
                x_offset=x_offset,
            ) for centroid_label in centroid_labels
        ]
        shifted_deviate_labels = [
            deviate_label.to_shifted_page_char_regression_label(
                y_offset=y_offset,
                x_offset=x_offset,
            ) for deviate_label in deviate_labels
        ]

        # Crop image and score map.
        page_image = cropper.crop_image(page_image)
        page_score_map = cropper.crop_score_map(
            page_score_map,
            core_only=True,
        )

        return CroppedPageTextRegion(
            page_image=page_image,
            page_score_map=page_score_map,
            page_char_regression_labels=shifted_centroid_labels + shifted_deviate_labels,
            core_box=cropper.core_box,
        )

    def run(self, input: PageTextRegionCroppingStepInput, rng: RandomGenerator):
        page_cropping_step_output = input.page_cropping_step_output
        num_cropped_pages = len(page_cropping_step_output.cropped_pages)

        page_text_region_step_output = input.page_text_region_step_output
        page_image = page_text_region_step_output.page_image
        shape_before_rotate = page_text_region_step_output.shape_before_rotate
        rotate_angle = page_text_region_step_output.rotate_angle

        page_text_region_label_step_output = input.page_text_region_label_step_output
        page_score_map = page_text_region_label_step_output.page_score_map
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
                page_score_map=page_score_map,
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
