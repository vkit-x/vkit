import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import sample_cv_resize_interpolation
from vkit.element import Mask, ScoreMap, Image
from .page_distortion import PageDistortionStep
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)


@attrs.define
class PageResizingStepConfig:
    resized_text_line_height_min: float = 1.5
    resized_text_line_height_max: float = 10.0


@attrs.define
class PageResizingStepOutput:
    page_image: Image
    page_char_mask: Mask
    page_char_height_score_map: ScoreMap
    page_text_line_mask: Mask
    page_text_line_height_score_map: ScoreMap


class PageResizingStep(
    PipelineStep[
        PageResizingStepConfig,
        PageResizingStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageResizingStepConfig):
        super().__init__(config)

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_distortion_step_output = state.get_pipeline_step_output(PageDistortionStep)
        page_image = page_distortion_step_output.page_image

        page_char_mask = page_distortion_step_output.page_char_mask
        assert page_char_mask

        page_char_height_score_map = page_distortion_step_output.page_char_height_score_map
        assert page_char_height_score_map

        page_text_line_mask = page_distortion_step_output.page_text_line_mask
        assert page_text_line_mask

        page_text_line_height_score_map = \
            page_distortion_step_output.page_text_line_height_score_map
        assert page_text_line_height_score_map

        page_distorted_text_line_heights = page_distortion_step_output.page_text_line_heights
        assert page_distorted_text_line_heights

        # Resizing.
        height, width = page_image.shape
        text_line_heights_min = min(page_distorted_text_line_heights)
        resized_text_line_height = rng.uniform(
            self.config.resized_text_line_height_min,
            self.config.resized_text_line_height_max,
        )
        resize_ratio = resized_text_line_height / text_line_heights_min

        resized_height = round(resize_ratio * height)
        resized_width = round(resize_ratio * width)

        cv_resize_interpolation = sample_cv_resize_interpolation(
            rng,
            include_cv_inter_area=(resize_ratio < 1.0),
        )

        page_image = page_image.to_resized_image(
            resized_height=resized_height,
            resized_width=resized_width,
            cv_resize_interpolation=cv_resize_interpolation,
        )

        assert page_char_mask.shape == (height, width)
        page_char_mask = page_char_mask.to_resized_mask(
            resized_height=resized_height,
            resized_width=resized_width,
            cv_resize_interpolation=cv_resize_interpolation,
        )

        assert page_char_height_score_map.shape == (height, width)
        page_char_height_score_map = page_char_height_score_map.to_resized_score_map(
            resized_height=resized_height,
            resized_width=resized_width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        # Scores are resized as well.
        page_char_height_score_map.mat *= resize_ratio

        assert page_text_line_mask.shape == (height, width)
        page_text_line_mask = page_text_line_mask.to_resized_mask(
            resized_height=resized_height,
            resized_width=resized_width,
            cv_resize_interpolation=cv_resize_interpolation,
        )

        assert page_text_line_height_score_map.shape == (height, width)
        page_text_line_height_score_map = page_text_line_height_score_map.to_resized_score_map(
            resized_height=resized_height,
            resized_width=resized_width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        # Scores are resized as well.
        page_text_line_height_score_map.mat *= resize_ratio

        return PageResizingStepOutput(
            page_image=page_image,
            page_char_mask=page_char_mask,
            page_char_height_score_map=page_char_height_score_map,
            page_text_line_mask=page_text_line_mask,
            page_text_line_height_score_map=page_text_line_height_score_map,
        )


page_resizing_step_factory = PipelineStepFactory(PageResizingStep)
