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
from typing import Optional, Sequence
from enum import unique, Enum

import numpy as np
import numpy.typing as npt
import attrs

from vkit.element import Image, ImageMode


def extract_mat_from_image(
    image: Image,
    dtype: npt.DTypeLike,
    channels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    mat = image.mat
    if channels:
        mat: np.ndarray = mat[:, :, channels]
    return mat.astype(dtype)


@unique
class OutOfBoundBehavior(Enum):
    CLIP = 'clip'
    CYCLE = 'cycle'


def clip_mat_back_to_uint8(mat: np.ndarray) -> np.ndarray:
    return np.clip(mat, 0, 255).astype(np.uint8)


def cycle_mat_back_to_uint8(mat: np.ndarray) -> np.ndarray:
    return (mat % 256).astype(np.uint8)


def handle_out_of_bound_and_dtype(mat: np.ndarray, oob_behavior: OutOfBoundBehavior):
    mat = np.round(mat)
    if oob_behavior == OutOfBoundBehavior.CLIP:
        mat = clip_mat_back_to_uint8(mat)
    elif oob_behavior == OutOfBoundBehavior.CYCLE:
        mat = cycle_mat_back_to_uint8(mat)
    else:
        raise NotImplementedError()
    return mat


def generate_new_image(
    image: Image,
    new_mat: np.ndarray,
    channels: Optional[Sequence[int]] = None,
):
    if channels:
        new_image = image.copy()
        with new_image.writable_context:
            new_image.mat[:, :, channels] = new_mat
    else:
        assert image.mat.shape == new_mat.shape
        new_image = attrs.evolve(image, mat=new_mat)

    return new_image


def to_rgb_image(image: Image, mode: ImageMode):
    if mode not in (ImageMode.GRAYSCALE, ImageMode.RGB):
        image = image.to_rgb_image()
    return image


def to_original_image(image: Image, mode: ImageMode):
    if mode not in (ImageMode.GRAYSCALE, ImageMode.RGB):
        image = image.to_target_mode_image(mode)
    return image
