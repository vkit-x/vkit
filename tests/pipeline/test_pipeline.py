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
from typing import List, Set, Sequence
import functools

import attrs
from numpy.random import Generator as RandomGenerator, default_rng

from vkit.element import Point, Line, Box, Polygon, Image
from vkit.mechanism.painter import Painter
from vkit.pipeline import (
    pipeline_step_collection_factory,
    PageDistortionStepOutput,
    PageResizingStepOutput,
    PageCroppingStepOutput,
    PageTextRegionStepOutput,
    PageCharRegressionLabel,
    PageCharRegressionLabelTag,
    PageTextRegionLabelStepOutput,
    PageTextRegionCroppingStepOutput,
    PipelinePostProcessor,
    PipelinePostProcessorFactory,
    Pipeline,
)
from vkit.pipeline.text_detection.page_text_region import build_background_image_for_stacking
from tests.opt import write_image


@attrs.define
class DebugAdaptiveScalingPipelinePostProcessorConfig:
    pass


@attrs.define
class DebugAdaptiveScalingPipelinePostProcessorInputOutput:
    page_distortion_step_output: PageDistortionStepOutput
    page_resizing_step_output: PageResizingStepOutput
    page_cropping_step_output: PageCroppingStepOutput
    page_text_region_step_output: PageTextRegionStepOutput
    page_text_region_label_step_output: PageTextRegionLabelStepOutput
    page_text_region_cropping_step_output: PageTextRegionCroppingStepOutput


class DebugAdaptiveScalingPipelinePostProcessor(
    PipelinePostProcessor[
        DebugAdaptiveScalingPipelinePostProcessorConfig,
        DebugAdaptiveScalingPipelinePostProcessorInputOutput,
        DebugAdaptiveScalingPipelinePostProcessorInputOutput,
    ]
):  # yapf: disable

    def generate_output(
        self,
        input: DebugAdaptiveScalingPipelinePostProcessorInputOutput,
        rng: RandomGenerator,
    ):
        return input


def visualize_page_distortion_step_output(seed: int, output: PageDistortionStepOutput):
    cur_write_image = functools.partial(write_image, frames_offset=1)

    # Image.
    cur_write_image(f'page_{seed}.jpg', output.page_image)

    # Active mask.
    image = output.page_image.copy()
    output.page_active_mask.fill_image(image, (255, 0, 0), 0.5)
    cur_write_image(f'page_{seed}_active_mask.jpg', image)

    # Char-level polygon ordering.
    if False:
        points: List[Point] = []
        point_colors: List[str] = []
        for polygon in output.page_char_polygon_collection.char_polygons:
            assert polygon.num_points == 4
            points.extend(polygon.points)
            point_colors.extend(['red', 'green', 'blue', 'lightskyblue'])
        painter = Painter.create(output.page_image)
        painter.paint_points(points, color=point_colors, radius=1)
        cur_write_image(
            f'page_{seed}_char_polygons.png',
            painter.image,
            frames_offset=1,
        )

    # Char-level seal impression polygons.
    if False:
        painter = Painter.create(output.page_image)
        painter.paint_polygons(output.page_seal_impression_char_polygon_collection.char_polygons)
        cur_write_image(
            f'page_{seed}_seal_impression_char_polygons.png',
            painter.image,
            frames_offset=1,
        )

    # Char level mask.
    image = output.page_image.copy()
    page_distorted_char_mask = output.page_char_mask
    assert page_distorted_char_mask
    page_distorted_char_mask.fill_image(image, (255, 0, 0), 0.5)

    if False:
        points: List[Point] = []
        point_colors: List[str] = []
        for polygon in output.page_char_polygon_collection.char_polygons:
            assert polygon.num_points == 4
            points.extend(polygon.points)
            point_colors.extend(['red', 'green', 'blue', 'lightskyblue'])
        painter = Painter.create(image)
        painter.paint_points(points, color=point_colors, radius=1, alpha=1.0)
        image = painter.image

    cur_write_image(f'page_{seed}_char_mask.jpg', image, frames_offset=1)

    # Char level score map.
    painter = Painter.create(output.page_image)
    page_distorted_char_height_score_map = output.page_char_height_score_map
    assert page_distorted_char_height_score_map
    painter.paint_score_map(page_distorted_char_height_score_map)
    cur_write_image(f'page_{seed}_char_score_map.jpg', painter.image)

    # Text line level mask.
    image = output.page_image.copy()
    page_distorted_text_line_mask = output.page_text_line_mask
    assert page_distorted_text_line_mask
    page_distorted_text_line_mask.fill_image(image, (255, 0, 0), 0.5)
    cur_write_image(f'page_{seed}_text_line_mask.jpg', image)

    # Text line level score map.
    painter = Painter.create(output.page_image)
    page_distorted_text_line_height_score_map = output.page_text_line_height_score_map
    assert page_distorted_text_line_height_score_map
    painter.paint_score_map(page_distorted_text_line_height_score_map)
    cur_write_image(f'page_{seed}_text_line_score_map.jpg', painter.image)

    if True:
        painter = Painter.create(output.page_image)
        painter.paint_polygons(output.page_disconnected_text_region_collection.to_polygons())
        cur_write_image(f'page_{seed}_disconnected_text_region_collection.jpg', painter.image)

    if True:
        painter = Painter.create(output.page_image)
        painter.paint_polygons(output.page_non_text_region_collection.to_polygons())
        cur_write_image(f'page_{seed}_non_text_region_collection.jpg', painter.image)


def visualize_page_cropping_step_output(seed: int, output: PageCroppingStepOutput):
    cur_write_image = functools.partial(write_image, frames_offset=1)

    for idx, cropped_page in enumerate(output.cropped_pages[:3]):
        cur_write_image(f'page_{seed}_cropped_{idx}_image.png', cropped_page.page_image)
        painter = Painter.create(cropped_page.page_image)
        painter.paint_mask(cropped_page.page_char_mask)
        cur_write_image(f'page_{seed}_cropped_{idx}_mask.png', painter.image)

        painter = Painter.create(cropped_page.page_image)
        painter.paint_mask(cropped_page.page_seal_impression_char_mask)
        cur_write_image(f'page_{seed}_cropped_{idx}_seal_impression_mask.png', painter.image)

        painter = Painter.create(cropped_page.page_image)
        painter.paint_score_map(cropped_page.page_char_height_score_map)
        cur_write_image(f'page_{seed}_cropped_{idx}_score_map.png', painter.image)

        assert cropped_page.page_char_mask.box
        char_mask_area = cropped_page.page_char_mask.box.area
        char_mask_ratio = (cropped_page.page_char_mask.mat > 0).sum() / char_mask_area
        print(f'{idx}: char_mask_ratio = {char_mask_ratio}')

        downsampled_label = cropped_page.downsampled_label
        assert downsampled_label

        page_downsampled_char_mask = downsampled_label.page_char_mask
        assert page_downsampled_char_mask
        page_downsampled_char_height_score_map = downsampled_label.page_char_height_score_map
        assert page_downsampled_char_height_score_map

        painter = Painter.create(page_downsampled_char_mask)
        painter.paint_mask(page_downsampled_char_mask)
        cur_write_image(f'page_{seed}_cropped_{idx}_ds_mask.png', painter.image)

        painter = Painter.create(page_downsampled_char_height_score_map)
        painter.paint_score_map(page_downsampled_char_height_score_map)
        cur_write_image(f'page_{seed}_cropped_{idx}_ds_score_map.png', painter.image)


def visualize_page_resizing_step_output(seed: int, output: PageResizingStepOutput):
    cur_write_image = functools.partial(write_image, frames_offset=1)

    cur_write_image(f'page_{seed}_resized_image.jpg', output.page_image)

    painter = Painter(output.page_image)
    painter.paint_mask(output.page_seal_impression_char_mask)
    cur_write_image(f'page_{seed}_resized_seal_impression_char_mask.jpg', painter.image)

    painter = Painter(output.page_image)
    painter.paint_mask(output.page_text_line_mask, alpha=0.9)
    cur_write_image(f'page_{seed}_resized_text_line_mask.jpg', painter.image)

    if False:
        text_line_disconnected_masks = \
            output.page_text_line_mask.to_disconnected_polygon_mask_pairs()
        painter = Painter(output.page_image)
        painter.paint_masks(text_line_disconnected_masks, alpha=0.9)
        cur_write_image(f'page_{seed}_resized_text_line_disconnected_masks.jpg', painter.image)

        text_line_disconnected_polygons = output.page_text_line_mask.to_disconnected_polygons()
        painter = Painter(output.page_image)
        painter.paint_polygons(text_line_disconnected_polygons, alpha=0.9)
        cur_write_image(f'page_{seed}_resized_text_line_disconnected_polygons.jpg', painter.image)


def visualize_page_text_region_step_output(
    seed: int,
    output: PageTextRegionStepOutput,
):
    cur_write_image = functools.partial(write_image, frames_offset=1)

    if output.debug:
        precise_text_region_polygons: List[Polygon] = []
        char_polygons: List[Polygon] = []
        color_indices: List[int] = []
        for color_idx, page_text_region_info in enumerate(output.debug.page_text_region_infos):
            precise_text_region_polygons.append(page_text_region_info.precise_text_region_polygon)
            char_polygons.extend(page_text_region_info.char_polygons)
            color_indices.extend([color_idx] * len(page_text_region_info.char_polygons))

        painter = Painter.create(output.debug.page_image)
        painter.paint_polygons(precise_text_region_polygons)
        cur_write_image(f'page_{seed}_precise_text_region_polygons.jpg', painter.image)

        painter = Painter.create(output.debug.page_image)
        painter.paint_polygons(char_polygons, color=color_indices)
        cur_write_image(f'page_{seed}_precise_text_region_char_polygons.jpg', painter.image)

        for idx, flattened_text_region in enumerate(output.debug.flattened_text_regions[:3]):
            cur_write_image(
                f'page_{seed}_flat_text_region_{idx}.jpg',
                flattened_text_region.flattened_image,
            )

            painter = Painter.create(flattened_text_region.flattened_image)
            assert flattened_text_region.flattened_char_polygons
            painter.paint_polygons(
                flattened_text_region.flattened_char_polygons,
                enable_polygon_points=True,
            )
            cur_write_image(f'page_{seed}_flat_text_region_{idx}_char_polygons.jpg', painter.image)

        if False:
            for idx, flattened_text_region in enumerate(output.debug.flattened_text_regions):
                cur_write_image(
                    f'page_{seed}_flat_text_region_debug_{idx}_original.jpg',
                    flattened_text_region.text_region_image,
                )
                cur_write_image(
                    f'page_{seed}_flat_text_region_debug_{idx}_flattened.jpg',
                    flattened_text_region.flattened_image,
                )

    cur_write_image(f'page_{seed}_stacked_image.jpg', output.page_image)

    painter = Painter.create(output.page_image)
    painter.paint_polygons(output.page_char_polygons)
    cur_write_image(f'page_{seed}_stacked_image_char_polygons.jpg', painter.image)


def paint_page_char_regression_labels_way_0(
    page_image: Image,
    page_char_regression_labels: Sequence[PageCharRegressionLabel],
):
    points: List[Point] = []
    points_color: List[str] = []
    visited_char_indices: Set[int] = set()

    lines: List[Line] = []
    lines_color: List[str] = []

    for label in page_char_regression_labels:
        label_point = Point.create(y=label.label_point_smooth_y, x=label.label_point_smooth_x)
        points.append(label_point)
        if label.tag == PageCharRegressionLabelTag.CENTROID:
            points_color.append('red')
        elif label.tag == PageCharRegressionLabelTag.DEVIATE:
            points_color.append('green')
        else:
            raise NotImplementedError()

        if label.char_idx not in visited_char_indices:
            points.extend([label.up_left, label.up_right, label.down_right, label.down_left])
            points_color.extend(['blue', 'yellow', 'white', '#00ffff'])
            visited_char_indices.add(label.char_idx)

        lines.extend([
            Line(point_begin=label_point, point_end=label.up_left),
            Line(point_begin=label_point, point_end=label.up_right),
            Line(point_begin=label_point, point_end=label.down_right),
            Line(point_begin=label_point, point_end=label.down_left),
        ])
        lines_color.extend(['blue', 'yellow', 'white', '#00ffff'])

    painter = Painter(page_image)
    painter.paint_lines(lines, color=lines_color, alpha=0.8)
    painter.paint_points(points, radius=2, color=points_color, alpha=0.9)
    return painter.image


def paint_page_char_regression_labels_way_1(
    page_image: Image,
    page_char_regression_labels: Sequence[PageCharRegressionLabel],
):
    visited_char_indices: Set[int] = set()
    points: List[Point] = []
    points_color: List[str] = []
    lines: List[Line] = []
    lines_color: List[str] = []

    for label in page_char_regression_labels:
        if label.char_idx in visited_char_indices:
            continue

        points.extend([label.up_left, label.up_right, label.down_right, label.down_left])
        points_color.extend(['blue', 'yellow', 'white', '#00ffff'])
        visited_char_indices.add(label.char_idx)

        bounding_up_left = Point.create(y=label.bounding_smooth_up, x=label.bounding_smooth_left)
        bounding_up_right = Point.create(y=label.bounding_smooth_up, x=label.bounding_smooth_right)
        bounding_down_left = Point.create(
            y=label.bounding_smooth_down,
            x=label.bounding_smooth_left,
        )
        bounding_down_right = Point.create(
            y=label.bounding_smooth_down,
            x=label.bounding_smooth_right,
        )

        lines.append(Line(point_begin=bounding_up_left, point_end=bounding_up_right))
        lines.append(Line(point_begin=bounding_down_left, point_end=bounding_down_right))
        lines.append(Line(point_begin=bounding_up_left, point_end=bounding_down_left))
        lines.append(Line(point_begin=bounding_up_right, point_end=bounding_down_right))
        color = ['blue'] * 4
        color[label.bounding_orientation_idx] = 'red'
        lines_color.extend(color)

    painter = Painter(page_image)
    painter.paint_lines(lines, color=lines_color, alpha=0.8)
    painter.paint_points(points, radius=2, color=points_color, alpha=0.9)
    return painter.image


def visualize_page_text_region_label_step_output(
    seed: int,
    page_image: Image,
    output: PageTextRegionLabelStepOutput,
):
    cur_write_image = functools.partial(write_image, frames_offset=1)

    painter = Painter(page_image)
    painter.paint_score_map(output.page_char_gaussian_score_map)
    cur_write_image(f'page_{seed}_stacked_image_label_char_score_map.jpg', painter.image)

    def point_distance(point0: Point, point1: Point):
        import math
        return math.hypot(point0.smooth_y - point1.smooth_y, point0.smooth_x - point1.smooth_x)

    def check_point_reconstruction(label: PageCharRegressionLabel):
        import math
        import numpy as np

        label_point = Point.create(y=label.label_point_smooth_y, x=label.label_point_smooth_x)

        offset_y, offset_x = label.generate_up_left_offsets()
        up_left = Point.create(y=label_point.smooth_y + offset_y, x=label_point.smooth_x + offset_x)
        assert math.isclose(point_distance(up_left, label.up_left), 0, abs_tol=1E-3)

        theta = np.arctan2(offset_y, offset_x)
        two_pi = 2 * np.pi
        theta = theta % two_pi

        angle_distrib = label.generate_clockwise_angle_distribution()
        _, up_right_dis, down_right_dis, down_left_dis = label.generate_clockwise_distances()

        theta += angle_distrib[0] * two_pi
        theta = theta % two_pi
        up_right = Point.create(
            y=label_point.smooth_y + np.sin(theta) * up_right_dis,
            x=label_point.smooth_x + np.cos(theta) * up_right_dis,
        )
        assert math.isclose(point_distance(up_right, label.up_right), 0, abs_tol=1E-3)

        theta += angle_distrib[1] * two_pi
        theta = theta % two_pi
        down_right = Point.create(
            y=label_point.smooth_y + np.sin(theta) * down_right_dis,
            x=label_point.smooth_x + np.cos(theta) * down_right_dis,
        )
        assert math.isclose(point_distance(down_right, label.down_right), 0, abs_tol=1E-3)

        theta += angle_distrib[2] * two_pi
        theta = theta % two_pi
        down_left = Point.create(
            y=label_point.smooth_y + np.sin(theta) * down_left_dis,
            x=label_point.smooth_x + np.cos(theta) * down_left_dis,
        )
        assert math.isclose(point_distance(down_left, label.down_left), 0, abs_tol=1E-3)

    for label in output.page_char_regression_labels:
        check_point_reconstruction(label)

    cur_write_image(
        f'page_{seed}_stacked_image_label_char_regression_way_0.jpg',
        paint_page_char_regression_labels_way_0(page_image, output.page_char_regression_labels),
    )

    cur_write_image(
        f'page_{seed}_stacked_image_label_char_regression_way_1.jpg',
        paint_page_char_regression_labels_way_1(page_image, output.page_char_regression_labels),
    )


def paint_image_cell(
    image: Image,
    downsampling_factor: int,
):
    assert downsampling_factor > 1
    ups = list(range(0, image.height, downsampling_factor))
    lefts = list(range(0, image.width, downsampling_factor))
    boxes: List[Box] = []
    color: List[str] = []
    for up_idx, up in enumerate(ups):
        down = up + downsampling_factor - 1
        if down >= image.height:
            break
        for left_idx, left in enumerate(lefts):
            right = left + downsampling_factor - 1
            if right >= image.width:
                break
            boxes.append(Box(up=up, down=down, left=left, right=right))
            color.append(['red', 'blue'][(up_idx + left_idx) % 2])

    painter = Painter(image)
    painter.paint_boxes(boxes, color=color, alpha=0.3)
    return painter.image


def visualize_page_text_region_cropping_step_output(
    seed: int,
    output: PageTextRegionCroppingStepOutput,
):
    cur_write_image = functools.partial(write_image, frames_offset=1)

    for idx, cropped_page_text_region in enumerate(output.cropped_page_text_regions[:3]):
        cur_write_image(
            f'page_{seed}_cropped_text_region_{idx}_image.jpg',
            cropped_page_text_region.page_image,
        )

        if False:
            cur_write_image(
                f'page_{seed}_cropped_text_region_{idx}_image_ds_8.jpg',
                paint_image_cell(cropped_page_text_region.page_image, 8),
            )
            cur_write_image(
                f'page_{seed}_cropped_text_region_{idx}_image_ds_16.jpg',
                paint_image_cell(cropped_page_text_region.page_image, 16),
            )
            cur_write_image(
                f'page_{seed}_cropped_text_region_{idx}_image_ds_32.jpg',
                paint_image_cell(cropped_page_text_region.page_image, 32),
            )

        painter = Painter(cropped_page_text_region.page_image)
        painter.paint_mask(cropped_page_text_region.page_char_mask)
        cur_write_image(
            f'page_{seed}_cropped_text_region_{idx}_mask.jpg',
            painter.image,
        )

        painter = Painter(cropped_page_text_region.page_image)
        painter.paint_score_map(cropped_page_text_region.page_char_height_score_map)
        cur_write_image(
            f'page_{seed}_cropped_text_region_{idx}_height_score_map.jpg',
            painter.image,
        )

        painter = Painter(cropped_page_text_region.page_image)
        painter.paint_score_map(cropped_page_text_region.page_char_gaussian_score_map)
        cur_write_image(
            f'page_{seed}_cropped_text_region_{idx}_gaussian_score_map.jpg',
            painter.image,
        )

        cur_write_image(
            f'page_{seed}_cropped_text_region_{idx}_char_regression_label_way_0.jpg',
            paint_page_char_regression_labels_way_0(
                cropped_page_text_region.page_image,
                cropped_page_text_region.page_char_regression_labels,
            ),
        )

        cur_write_image(
            f'page_{seed}_cropped_text_region_{idx}_char_regression_label_way_1.jpg',
            paint_page_char_regression_labels_way_1(
                cropped_page_text_region.page_image,
                cropped_page_text_region.page_char_regression_labels,
            ),
        )

        painter = Painter(cropped_page_text_region.page_image)
        painter.paint_mask(cropped_page_text_region.page_char_bounding_box_mask)
        cur_write_image(
            f'page_{seed}_cropped_text_region_{idx}_char_bounding_box_mask.jpg',
            painter.image,
        )

        downsampled_label = cropped_page_text_region.downsampled_label
        assert downsampled_label

        page_downsampled_char_mask = downsampled_label.page_char_mask
        assert page_downsampled_char_mask
        page_downsampled_char_height_score_map = downsampled_label.page_char_height_score_map
        assert page_downsampled_char_height_score_map

        painter = Painter.create(page_downsampled_char_mask)
        painter.paint_mask(page_downsampled_char_mask)
        cur_write_image(f'page_{seed}_cropped_text_region_{idx}_ds_mask.png', painter.image)

        painter = Painter.create(page_downsampled_char_height_score_map)
        painter.paint_score_map(page_downsampled_char_height_score_map)
        cur_write_image(f'page_{seed}_cropped_text_region_{idx}_ds_score_map.png', painter.image)


def debug_adaptive_scaling_dataset_steps():
    post_processor_factory = PipelinePostProcessorFactory(DebugAdaptiveScalingPipelinePostProcessor)
    pipeline_config_json = '$VKIT_DATA/pipeline/debug_adaptive_scaling_dataset_steps.json'
    # pipeline_config_json = '$VKIT_ARTIFACT_PACK/pipeline/text_detection/dev_adaptive_scaling_dataset_steps.json'  # noqa
    pipeline = Pipeline(
        steps=pipeline_step_collection_factory.create(pipeline_config_json),
        post_processor=post_processor_factory.create(),
    )
    for seed in range(10):
        print(seed)
        rng = default_rng(seed)
        output = pipeline.run(rng)
        visualize_page_distortion_step_output(seed, output.page_distortion_step_output)
        if True:
            visualize_page_resizing_step_output(seed, output.page_resizing_step_output)
        if True:
            visualize_page_cropping_step_output(seed, output.page_cropping_step_output)
        if True:
            visualize_page_text_region_step_output(
                seed,
                output.page_text_region_step_output,
            )
        if True:
            visualize_page_text_region_label_step_output(
                seed,
                output.page_text_region_step_output.page_image,
                output.page_text_region_label_step_output,
            )
        if True:
            visualize_page_text_region_cropping_step_output(
                seed,
                output.page_text_region_cropping_step_output,
            )


def profile_adaptive_scaling_dataset_steps():
    post_processor_factory = PipelinePostProcessorFactory(DebugAdaptiveScalingPipelinePostProcessor)
    pipeline = Pipeline(
        steps=pipeline_step_collection_factory.
        create('$VKIT_DATA/pipeline/debug_adaptive_scaling_dataset_steps.json'),
        post_processor=post_processor_factory.create(),
    )
    for seed in (0, 1):
        rng = default_rng(seed)
        pipeline.run(rng)


def test_build_background_image_for_stacking():
    image = build_background_image_for_stacking(300, 300)
    write_image('image.png', image)
