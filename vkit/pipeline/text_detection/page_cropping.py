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
from typing import Sequence, List, Optional, Tuple

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv

from vkit.element import Box, Mask, ScoreMap, Image, Cropper
from .page_resizing import PageResizingStepOutput
from ..interface import PipelineStep, PipelineStepFactory


@attrs.define
class PageCroppingStepConfig:
    core_size: int
    pad_size: int
    num_samples: Optional[int] = None
    num_samples_max: Optional[int] = None
    num_samples_estimation_factor: float = 1.5
    pad_value: int = 0
    drop_cropped_page_with_small_text_ratio: bool = True
    text_ratio_min: float = 0.025
    drop_cropped_page_with_large_black_area: bool = True
    black_area_ratio_max: float = 0.5
    enable_downsample_labeling: bool = True
    downsample_labeling_factor: int = 2

    @property
    def crop_size(self):
        return 2 * self.pad_size + self.core_size


@attrs.define
class PageCroppingStepInput:
    page_resizing_step_output: PageResizingStepOutput


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
        PageCroppingStepInput,
        PageCroppingStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageCroppingStepConfig):
        super().__init__(config)

    def sample_cropped_page(
        self,
        page_image: Image,
        page_char_mask: Mask,
        page_char_height_score_map: ScoreMap,
        page_text_line_mask: Mask,
        page_text_line_height_score_map: ScoreMap,
        rng: RandomGenerator,
    ):
        cropper = Cropper.create(
            shape=page_image.shape,
            core_size=self.config.core_size,
            pad_size=self.config.pad_size,
            pad_value=self.config.pad_value,
            rng=rng,
        )

        page_image = cropper.crop_image(page_image)

        page_char_mask = cropper.crop_mask(
            page_char_mask,
            core_only=True,
        )
        page_char_height_score_map = cropper.crop_score_map(
            page_char_height_score_map,
            core_only=True,
        )

        page_text_line_mask = cropper.crop_mask(
            page_text_line_mask,
            core_only=True,
        )
        page_text_line_height_score_map = cropper.crop_score_map(
            page_text_line_height_score_map,
            core_only=True,
        )

        if self.config.drop_cropped_page_with_small_text_ratio:
            num_text_pixels = (page_char_mask.mat > 0).sum()
            text_ratio = num_text_pixels / cropper.core_box.area
            if text_ratio < self.config.text_ratio_min:
                return None

        if self.config.drop_cropped_page_with_large_black_area:
            black_pixel_count = int((np.amax(page_image.mat, axis=2) == 0).sum())
            black_area_ratio = black_pixel_count / (page_image.height * page_image.width)
            if black_area_ratio >= self.config.black_area_ratio_max:
                return None

        downsampled_label: Optional[DownsampledLabel] = None
        if self.config.enable_downsample_labeling:
            downsample_labeling_factor = self.config.downsample_labeling_factor

            assert self.config.crop_size % downsample_labeling_factor == 0
            downsampled_size = self.config.crop_size // downsample_labeling_factor
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
            core_box=cropper.core_box,
            downsampled_label=downsampled_label,
        )

    def run(self, input: PageCroppingStepInput, rng: RandomGenerator):
        page_resizing_step_output = input.page_resizing_step_output
        page_image = page_resizing_step_output.page_image
        page_char_mask = page_resizing_step_output.page_char_mask
        page_char_height_score_map = page_resizing_step_output.page_char_height_score_map
        page_text_line_mask = page_resizing_step_output.page_text_line_mask
        page_text_line_height_score_map = page_resizing_step_output.page_text_line_height_score_map

        num_samples = self.config.num_samples

        if num_samples is None:
            page_image_area = int((np.amax(page_image.mat, axis=2) > 0).sum())
            core_area = self.config.core_size**2
            num_samples = max(
                1,
                round(page_image_area / core_area * self.config.num_samples_estimation_factor),
            )

        if self.config.num_samples_max:
            num_samples = min(num_samples, self.config.num_samples_max)

        run_count_max = max(3, 2 * num_samples)
        run_count = 0

        cropped_pages: List[CroppedPage] = []

        while len(cropped_pages) < num_samples and run_count < run_count_max:
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
            run_count += 1

        return PageCroppingStepOutput(cropped_pages=cropped_pages)


page_cropping_step_factory = PipelineStepFactory(PageCroppingStep)
