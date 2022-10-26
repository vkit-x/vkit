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

from vkit.element import Box, Mask, ScoreMap, Image
from vkit.mechanism.cropper import Cropper
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
    drop_cropped_page_with_small_active_region: bool = True
    active_region_ratio_min: float = 0.4
    enable_downsample_labeling: bool = True
    downsample_labeling_factor: int = 2


@attrs.define
class PageCroppingStepInput:
    page_resizing_step_output: PageResizingStepOutput


@attrs.define
class DownsampledLabel:
    shape: Tuple[int, int]
    page_char_mask: Mask
    page_char_height_score_map: ScoreMap
    page_text_line_mask: Mask
    page_text_line_height_score_map: ScoreMap
    core_box: Box


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
        page_active_mask: Mask,
        page_char_mask: Mask,
        page_char_height_score_map: ScoreMap,
        page_text_line_mask: Mask,
        page_text_line_height_score_map: ScoreMap,
        rng: RandomGenerator,
        force_crop_center: bool = False,
    ):
        if not force_crop_center:
            cropper = Cropper.create(
                shape=page_image.shape,
                core_size=self.config.core_size,
                pad_size=self.config.pad_size,
                pad_value=self.config.pad_value,
                rng=rng,
            )
        else:
            cropper = Cropper.create_from_center_point(
                shape=page_image.shape,
                core_size=self.config.core_size,
                pad_size=self.config.pad_size,
                pad_value=self.config.pad_value,
                center_point=Box.from_shapable(page_image).get_center_point(),
            )

        page_image = cropper.crop_image(page_image)

        page_active_mask = cropper.crop_mask(page_active_mask)

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

        if self.config.drop_cropped_page_with_small_active_region:
            num_active_pixels = int(page_active_mask.np_mask.sum())
            active_region_ratio = num_active_pixels / page_image.area
            if active_region_ratio < self.config.active_region_ratio_min:
                return None

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

            downsampled_page_text_line_mask = page_text_line_mask.to_box_detached()
            downsampled_page_text_line_mask = \
                downsampled_page_text_line_mask.to_resized_mask(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_page_text_line_height_score_map = \
                page_text_line_height_score_map.to_box_detached()
            downsampled_page_text_line_height_score_map = \
                downsampled_page_text_line_height_score_map.to_resized_score_map(
                    resized_height=downsampled_core_size,
                    resized_width=downsampled_core_size,
                    cv_resize_interpolation=cv.INTER_AREA,
                )

            downsampled_label = DownsampledLabel(
                shape=downsampled_shape,
                page_char_mask=downsampled_page_char_mask,
                page_char_height_score_map=downsampled_page_char_height_score_map,
                page_text_line_mask=downsampled_page_text_line_mask,
                page_text_line_height_score_map=downsampled_page_text_line_height_score_map,
                core_box=downsampled_core_box,
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
        page_active_mask = page_resizing_step_output.page_active_mask
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
                page_active_mask=page_active_mask,
                page_char_mask=page_char_mask,
                page_char_height_score_map=page_char_height_score_map,
                page_text_line_mask=page_text_line_mask,
                page_text_line_height_score_map=page_text_line_height_score_map,
                rng=rng,
                force_crop_center=(run_count == 0),
            )
            if cropped_page:
                cropped_pages.append(cropped_page)
            run_count += 1

        return PageCroppingStepOutput(cropped_pages=cropped_pages)


page_cropping_step_factory = PipelineStepFactory(PageCroppingStep)
