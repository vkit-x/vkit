from typing import List
import functools

import attrs
from numpy.random import Generator as RandomGenerator, default_rng
import pytest

from vkit.element import Point, Painter
from vkit.pipeline import (
    pipeline_step_collection_factory,
    PageDistortionStepOutput,
    PageCroppingStepOutput,
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
    page_cropping_step_output: PageCroppingStepOutput


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
