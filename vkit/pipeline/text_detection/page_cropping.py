from typing import Sequence, List, Optional, Tuple

import attrs
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.element import Box, Mask, ScoreMap, Image
from .page_resizing import PageResizingStep
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)


@attrs.define
class PageCroppingStepConfig:
    core_size: int
    pad_size: int
    num_samples: Optional[int] = None
    num_samples_estimation_factor: float = 1.5
    pad_value: int = 0
    drop_cropped_page_with_no_text: bool = True
    enable_downsample_labeling: bool = True
    downsample_labeling_factor: int = 2

    @property
    def crop_size(self):
        return 2 * self.pad_size + self.core_size


@attrs.define
class DownsampledLabel:
    shape: Tuple[int, int]
    core_box: Box
    page_char_mask: Mask
    page_char_height_score_map: ScoreMap
    page_text_line_mask: Mask
    page_text_line_height_score_map: ScoreMap


@attrs.define
class CroppedPage:
    page_image: Image
    page_char_mask: Mask
    page_char_height_score_map: ScoreMap
    page_text_line_mask: Mask
    page_text_line_height_score_map: ScoreMap
    core_box: Box
    downsampled_label: Optional[DownsampledLabel]


@attrs.define
class PageCroppingStepOutput:
    cropped_pages: Sequence[CroppedPage]


class PageCroppingStep(
    PipelineStep[
        PageCroppingStepConfig,
        PageCroppingStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageCroppingStepConfig):
        super().__init__(config)

    def sample_cropped_positions(self, length: int, rng: RandomGenerator):
        if self.config.core_size <= length:
            core_begin = rng.integers(0, length - self.config.core_size + 1)
            begin = core_begin - self.config.pad_size
            offset = 0
            if begin < 0:
                offset = abs(begin)
                begin = 0

        else:
            begin = 0
            offset = self.config.pad_size
            offset += rng.integers(0, self.config.core_size - length + 1)

        end = min(length - 1, begin + (self.config.crop_size - offset) - 1)
        return offset, begin, end, end + 1 - begin

    def sample_cropped_page(
        self,
        page_image: Image,
        page_char_mask: Mask,
        page_char_height_score_map: ScoreMap,
        page_text_line_mask: Mask,
        page_text_line_height_score_map: ScoreMap,
        rng: RandomGenerator,
    ):
        height, width = page_image.shape
        assert page_text_line_mask.shape == (height, width)
        assert page_text_line_height_score_map.shape == (height, width)

        vert_offset, up, down, origin_height = self.sample_cropped_positions(height, rng)
        hori_offset, left, right, origin_width = self.sample_cropped_positions(width, rng)

        cropped_shape = (self.config.crop_size, self.config.crop_size)
        origin_box = Box(up=up, down=down, left=left, right=right)

        page_image = origin_box.extract_image(page_image)

        page_char_mask = origin_box.extract_mask(page_char_mask)
        page_char_height_score_map = origin_box.extract_score_map(page_char_height_score_map)

        page_text_line_mask = origin_box.extract_mask(page_text_line_mask)
        page_text_line_height_score_map = origin_box.extract_score_map(
            page_text_line_height_score_map
        )

        if origin_height == origin_width == self.config.crop_size:
            assert vert_offset == hori_offset == 0
            assert page_text_line_mask.shape == cropped_shape
            assert page_text_line_height_score_map.shape == cropped_shape

        else:
            target_box = Box(
                up=vert_offset,
                down=vert_offset + origin_height - 1,
                left=hori_offset,
                right=hori_offset + origin_width - 1,
            )
            assert target_box.down < self.config.crop_size
            assert target_box.right < self.config.crop_size

            new_page_image = Image.from_shape(
                cropped_shape,
                num_channels=page_image.num_channels,
                value=self.config.pad_value,
            )
            target_box.fill_image(new_page_image, page_image)
            page_image = new_page_image

            new_page_char_mask = Mask.from_shape(cropped_shape)
            target_box.fill_mask(new_page_char_mask, page_char_mask)
            page_char_mask = new_page_char_mask

            new_page_char_height_score_map = ScoreMap.from_shape(
                cropped_shape,
                is_prob=False,
            )
            target_box.fill_score_map(new_page_char_height_score_map, page_char_height_score_map)
            page_char_height_score_map = new_page_char_height_score_map

            new_page_text_line_mask = Mask.from_shape(cropped_shape)
            target_box.fill_mask(new_page_text_line_mask, page_text_line_mask)
            page_text_line_mask = new_page_text_line_mask

            new_page_text_line_height_score_map = ScoreMap.from_shape(
                cropped_shape,
                is_prob=False,
            )
            target_box.fill_score_map(
                new_page_text_line_height_score_map, page_text_line_height_score_map
            )
            page_text_line_height_score_map = new_page_text_line_height_score_map

        assert page_image.shape == cropped_shape

        core_begin = self.config.pad_size
        core_end = core_begin + self.config.core_size - 1
        core_box = Box(up=core_begin, down=core_end, left=core_begin, right=core_end)

        page_char_mask = core_box.extract_mask(page_char_mask)
        page_char_mask = page_char_mask.to_box_attached(core_box)

        page_char_height_score_map = core_box.extract_score_map(page_char_height_score_map)
        page_char_height_score_map = page_char_height_score_map.to_box_attached(core_box)

        page_text_line_mask = core_box.extract_mask(page_text_line_mask)
        page_text_line_mask = page_text_line_mask.to_box_attached(core_box)

        page_text_line_height_score_map = core_box.extract_score_map(
            page_text_line_height_score_map
        )
        page_text_line_height_score_map = page_text_line_height_score_map.to_box_attached(core_box)

        if self.config.drop_cropped_page_with_no_text:
            if not page_text_line_mask.mat.any():
                return None

        downsampled_label: Optional[DownsampledLabel] = None
        if self.config.enable_downsample_labeling:
            downsample_labeling_factor = self.config.downsample_labeling_factor

            assert self.config.crop_size % downsample_labeling_factor == 0
            downsampled_size = self.config.crop_size // downsample_labeling_factor
            downsampled_shape = (downsampled_size, downsampled_size)

            assert self.config.pad_size % downsample_labeling_factor == 0
            assert self.config.core_size % downsample_labeling_factor == 0
            assert core_box.height == core_box.width == self.config.core_size

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

            downsampled_page_char_mask = page_char_mask.copy()
            downsampled_page_char_mask.box = None
            downsampled_page_char_mask = \
                downsampled_page_char_mask.to_resized_mask(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_page_char_height_score_map = page_char_height_score_map.copy()
            downsampled_page_char_height_score_map.box = None
            downsampled_page_char_height_score_map = \
                downsampled_page_char_height_score_map.to_resized_score_map(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_page_text_line_mask = page_text_line_mask.copy()
            downsampled_page_text_line_mask.box = None
            downsampled_page_text_line_mask = \
                downsampled_page_text_line_mask.to_resized_mask(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_page_text_line_height_score_map = page_text_line_height_score_map.copy()
            downsampled_page_text_line_height_score_map.box = None
            downsampled_page_text_line_height_score_map = \
                downsampled_page_text_line_height_score_map.to_resized_score_map(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_label = DownsampledLabel(
                shape=downsampled_shape,
                core_box=downsampled_core_box,
                page_char_mask=downsampled_page_char_mask,
                page_char_height_score_map=downsampled_page_char_height_score_map,
                page_text_line_mask=downsampled_page_text_line_mask,
                page_text_line_height_score_map=downsampled_page_text_line_height_score_map,
            )

        return CroppedPage(
            page_image=page_image,
            page_char_mask=page_char_mask,
            page_char_height_score_map=page_char_height_score_map,
            page_text_line_mask=page_text_line_mask,
            page_text_line_height_score_map=page_text_line_height_score_map,
            core_box=core_box,
            downsampled_label=downsampled_label,
        )

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_resizing_step_output = state.get_pipeline_step_output(PageResizingStep)
        page_image = page_resizing_step_output.page_image
        page_char_mask = page_resizing_step_output.page_char_mask
        page_char_height_score_map = page_resizing_step_output.page_char_height_score_map
        page_text_line_mask = page_resizing_step_output.page_text_line_mask
        page_text_line_height_score_map = page_resizing_step_output.page_text_line_height_score_map

        num_samples = self.config.num_samples
        if num_samples is None:
            page_image_area = page_image.height * page_image.width
            core_area = self.config.core_size**2
            num_samples = max(
                1,
                round(page_image_area / core_area * self.config.num_samples_estimation_factor),
            )

        cropped_pages: List[CroppedPage] = []
        for _ in range(num_samples):
            cropped_page = self.sample_cropped_page(
                page_image=page_image,
                page_char_mask=page_char_mask,
                page_char_height_score_map=page_char_height_score_map,
                page_text_line_mask=page_text_line_mask,
                page_text_line_height_score_map=page_text_line_height_score_map,
                rng=rng,
            )
            if cropped_page:
                cropped_pages.append(cropped_page)
        return PageCroppingStepOutput(cropped_pages=cropped_pages)


page_cropping_step_factory = PipelineStepFactory(PageCroppingStep)
