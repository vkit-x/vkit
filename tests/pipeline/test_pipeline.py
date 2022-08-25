from typing import List
import functools

import attrs
from numpy.random import Generator as RandomGenerator, default_rng
import pytest

from vkit.element import Point, Polygon, Painter  # noqa
from vkit.pipeline import (
    pipeline_step_collection_factory,
    PageDistortionStepOutput,
    PageResizingStepOutput,
    PageCroppingStepOutput,
    PageTextRegionStepOutput,
    PipelinePostProcessor,
    PipelinePostProcessorFactory,
    Pipeline,
)
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

    # Char level polygon ordering.
    if False:
        points: List[Point] = []
        point_colors: List[str] = []
        for polygon in output.page_char_polygon_collection.polygons:
            assert len(polygon.points) == 4
            points.extend(polygon.points)
            point_colors.extend(['red', 'green', 'blue', 'lightskyblue'])
        painter = Painter.create(output.page_image)
        painter.paint_points(points, color=point_colors, radius=1)
        cur_write_image(f'page_{seed}_char_polygon.png', painter.image, frames_offset=1)

    # Char level mask.
    image = output.page_image.copy()
    page_distorted_char_mask = output.page_char_mask
    assert page_distorted_char_mask
    page_distorted_char_mask.fill_image(image, (255, 0, 0), 0.5)
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


def visualize_page_cropping_step_output(seed: int, output: PageCroppingStepOutput):
    cur_write_image = functools.partial(write_image, frames_offset=1)

    for idx, cropped_page in enumerate(output.cropped_pages):
        cur_write_image(f'page_{seed}_cropped_{idx}_image.png', cropped_page.page_image)
        painter = Painter.create(cropped_page.page_image)
        painter.paint_mask(cropped_page.page_char_mask)
        cur_write_image(f'page_{seed}_cropped_{idx}_mask.png', painter.image)
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

        for idx, page_flat_text_region in enumerate(output.debug.page_flat_text_regions[:3]):
            cur_write_image(f'page_{seed}_flat_text_region_{idx}.jpg', page_flat_text_region.image)

            painter = Painter.create(page_flat_text_region.image)
            painter.paint_polygons(page_flat_text_region.char_polygons)
            cur_write_image(f'page_{seed}_flat_text_region_{idx}_char_polygons.jpg', painter.image)

        for idx, page_flat_text_region in enumerate(
            output.debug.resized_page_flat_text_regions[:3]
        ):
            cur_write_image(
                f'page_{seed}_flat_text_region_resized_{idx}.jpg', page_flat_text_region.image
            )

            painter = Painter.create(page_flat_text_region.image)
            painter.paint_polygons(page_flat_text_region.char_polygons)
            cur_write_image(
                f'page_{seed}_flat_text_region_resized_{idx}_char_polygons.jpg', painter.image
            )

    cur_write_image(f'page_{seed}_stacked_image.jpg', output.page_image)

    painter = Painter.create(output.page_image)
    painter.paint_polygons(output.page_char_polygons)
    cur_write_image(f'page_{seed}_stacked_image_char_polygons.jpg', painter.image)


@pytest.mark.local
def test_debug_adaptive_scaling_dataset_steps():
    post_processor_factory = PipelinePostProcessorFactory(DebugAdaptiveScalingPipelinePostProcessor)
    pipeline = Pipeline(
        steps=pipeline_step_collection_factory.
        create('$VKIT_DATA/pipeline/debug_adaptive_scaling_dataset_steps.json'),
        post_processor=post_processor_factory.create(),
    )
    for seed in range(10):
        print(seed)
        rng = default_rng(seed)
        output = pipeline.run(rng)
        visualize_page_distortion_step_output(seed, output.page_distortion_step_output)
        if True:
            visualize_page_resizing_step_output(seed, output.page_resizing_step_output)
        if False:
            visualize_page_cropping_step_output(seed, output.page_cropping_step_output)
        if True:
            visualize_page_text_region_step_output(
                seed,
                output.page_text_region_step_output,
            )

    # For profiling.
    # for seed in (0, 1):
    #     rng = default_rng(seed)
    #     pipeline.run(rng)
