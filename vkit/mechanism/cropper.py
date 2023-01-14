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
from typing import Tuple

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import (
    Point,
    Box,
    Mask,
    ScoreMap,
    Image,
)


@attrs.define
class CropperState:
    height: int
    width: int
    pad_value: int
    crop_size: int
    original_box: Box
    target_box: Box
    target_core_box: Box
    original_core_box: Box

    @classmethod
    def sample_cropping_positions_along_axis(
        cls,
        core_size: int,
        pad_size: int,
        crop_size: int,
        length: int,
        rng: RandomGenerator,
    ):
        if core_size <= length:
            core_begin = rng.integers(0, length - core_size + 1)
            begin = core_begin - pad_size
            target_offset = 0
            if begin < 0:
                target_offset = abs(begin)
                begin = 0

        else:
            begin = 0
            target_offset = pad_size
            target_offset += rng.integers(0, core_size - length + 1)

        end = min(length - 1, begin + (crop_size - target_offset) - 1)
        return target_offset, begin, end

    @classmethod
    def sample_cropping_positions(
        cls,
        height: int,
        width: int,
        core_size: int,
        pad_size: int,
        crop_size: int,
        rng: RandomGenerator,
    ):
        (
            target_vert_offset,
            original_up,
            original_down,
        ) = cls.sample_cropping_positions_along_axis(
            core_size=core_size,
            pad_size=pad_size,
            crop_size=crop_size,
            length=height,
            rng=rng,
        )
        (
            target_hori_offset,
            original_left,
            original_right,
        ) = cls.sample_cropping_positions_along_axis(
            core_size=core_size,
            pad_size=pad_size,
            crop_size=crop_size,
            length=width,
            rng=rng,
        )
        return (
            target_vert_offset,
            original_up,
            original_down,
            target_hori_offset,
            original_left,
            original_right,
        )

    @classmethod
    def create_from_cropping_positions(
        cls,
        height: int,
        width: int,
        pad_size: int,
        pad_value: int,
        core_size: int,
        crop_size: int,
        target_vert_offset: int,
        original_up: int,
        original_down: int,
        target_hori_offset: int,
        original_left: int,
        original_right: int,
    ):
        original_box = Box(
            up=original_up,
            down=original_down,
            left=original_left,
            right=original_right,
        )

        target_box = Box(
            up=target_vert_offset,
            down=target_vert_offset + original_box.height - 1,
            left=target_hori_offset,
            right=target_hori_offset + original_box.width - 1,
        )

        target_core_begin = pad_size
        target_core_end = target_core_begin + core_size - 1
        target_core_box = Box(
            up=target_core_begin,
            down=target_core_end,
            left=target_core_begin,
            right=target_core_end,
        )

        original_core_box = Box(
            up=original_up + target_core_box.up - target_box.up,
            down=original_down + target_core_box.down - target_box.down,
            left=original_left + target_core_box.left - target_box.left,
            right=original_right + target_core_box.right - target_box.right,
        )

        return CropperState(
            height=height,
            width=width,
            pad_value=pad_value,
            crop_size=crop_size,
            original_box=original_box,
            target_box=target_box,
            target_core_box=target_core_box,
            original_core_box=original_core_box,
        )

    @classmethod
    def create_from_random_proposal(
        cls,
        shape: Tuple[int, int],
        core_size: int,
        pad_size: int,
        pad_value: int,
        rng: RandomGenerator,
    ):
        height, width = shape
        crop_size = 2 * pad_size + core_size
        (
            target_vert_offset,
            original_up,
            original_down,
            target_hori_offset,
            original_left,
            original_right,
        ) = cls.sample_cropping_positions(
            height=height,
            width=width,
            core_size=core_size,
            pad_size=pad_size,
            crop_size=crop_size,
            rng=rng,
        )
        return cls.create_from_cropping_positions(
            height=height,
            width=width,
            pad_size=pad_size,
            pad_value=pad_value,
            core_size=core_size,
            crop_size=crop_size,
            target_vert_offset=target_vert_offset,
            original_up=original_up,
            original_down=original_down,
            target_hori_offset=target_hori_offset,
            original_left=original_left,
            original_right=original_right,
        )

    @classmethod
    def create_from_center_point(
        cls,
        shape: Tuple[int, int],
        core_size: int,
        pad_size: int,
        pad_value: int,
        center_point: Point,
    ):
        height, width = shape
        crop_size = 2 * pad_size + core_size

        assert 0 <= center_point.y < height
        assert 0 <= center_point.x < width

        target_vert_offset = 0
        up = center_point.y - crop_size // 2
        down = up + crop_size - 1
        if up < 0:
            target_vert_offset = abs(up)
            up = 0
        down = min(height - 1, down)

        target_hori_offset = 0
        left = center_point.x - crop_size // 2
        right = left + crop_size - 1
        if left < 0:
            target_hori_offset = abs(left)
            left = 0
        right = min(width - 1, right)

        return CropperState.create_from_cropping_positions(
            height=height,
            width=width,
            pad_size=pad_size,
            pad_value=pad_value,
            core_size=core_size,
            crop_size=crop_size,
            target_vert_offset=target_vert_offset,
            original_up=up,
            original_down=down,
            target_hori_offset=target_hori_offset,
            original_left=left,
            original_right=right,
        )

    @property
    def need_post_filling(self):
        return (
            self.original_box.height != self.crop_size or self.original_box.width != self.crop_size
        )

    @property
    def cropped_shape(self):
        return (self.crop_size,) * 2


class Cropper:

    @classmethod
    def create_from_random_proposal(
        cls,
        shape: Tuple[int, int],
        core_size: int,
        pad_size: int,
        rng: RandomGenerator,
        pad_value: int = 0,
    ):
        cropper_state = CropperState.create_from_random_proposal(
            shape=shape,
            core_size=core_size,
            pad_size=pad_size,
            pad_value=pad_value,
            rng=rng,
        )
        return Cropper(cropper_state)

    @classmethod
    def create_from_center_point(
        cls,
        shape: Tuple[int, int],
        core_size: int,
        pad_size: int,
        center_point: Point,
        pad_value: int = 0,
    ):
        cropper_state = CropperState.create_from_center_point(
            shape=shape,
            core_size=core_size,
            pad_size=pad_size,
            pad_value=pad_value,
            center_point=center_point,
        )
        return Cropper(cropper_state)

    def __init__(self, cropper_state: CropperState):
        self.cropper_state = cropper_state

    @property
    def original_box(self):
        return self.cropper_state.original_box

    @property
    def target_box(self):
        return self.cropper_state.target_box

    @property
    def target_core_box(self):
        return self.cropper_state.target_core_box

    @property
    def original_core_box(self):
        return self.cropper_state.original_core_box

    @property
    def need_post_filling(self):
        return self.cropper_state.need_post_filling

    @property
    def crop_size(self):
        return self.cropper_state.crop_size

    @property
    def cropped_shape(self):
        return self.cropper_state.cropped_shape

    @property
    def pad_value(self):
        return self.cropper_state.pad_value

    def crop_mask(self, mask: Mask, core_only: bool = False):
        mask = self.original_box.extract_mask(mask)

        if self.need_post_filling:
            new_mask = Mask.from_shape(self.cropped_shape)
            self.target_box.fill_mask(new_mask, mask)
            mask = new_mask

        if core_only:
            mask = self.target_core_box.extract_mask(mask)
            mask = mask.to_box_attached(self.target_core_box)

        return mask

    def crop_score_map(self, score_map: ScoreMap, core_only: bool = False):
        score_map = self.original_box.extract_score_map(score_map)

        if self.need_post_filling:
            new_score_map = ScoreMap.from_shape(
                self.cropped_shape,
                is_prob=score_map.is_prob,
            )
            self.target_box.fill_score_map(new_score_map, score_map)
            score_map = new_score_map

        if core_only:
            score_map = self.target_core_box.extract_score_map(score_map)
            score_map = score_map.to_box_attached(self.target_core_box)

        return score_map

    def crop_image(self, image: Image):
        image = self.original_box.extract_image(image)

        if self.need_post_filling:
            new_image = Image.from_shape(
                self.cropped_shape,
                num_channels=image.num_channels,
                value=self.pad_value,
            )
            self.target_box.fill_image(new_image, image)
            image = new_image

        return image
