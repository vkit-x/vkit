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
import numpy as np
import cv2 as cv

from vkit.element import (
    Image,
    ImageMode,
    ScoreMap,
    Mask,
)
from .type import ImageGrid


def create_image_from_image_grid(image_grid: ImageGrid, image_mode: ImageMode):
    ndim = image_mode.to_ndim()
    if ndim == 2:
        shape = (image_grid.image_height, image_grid.image_width)
    elif ndim == 3:
        num_channels = image_mode.to_num_channels()
        assert num_channels
        shape = (image_grid.image_height, image_grid.image_width, num_channels)
    else:
        raise NotImplementedError()

    dtype = image_mode.to_dtype()
    mat = np.zeros(shape, dtype=dtype)
    return Image(mat=mat, mode=image_mode)


def create_score_map_from_image_grid(image_grid: ImageGrid):
    shape = (image_grid.image_height, image_grid.image_width)
    mat = np.zeros(shape, dtype=np.float32)
    return ScoreMap(mat=mat)


def create_mask_from_image_grid(image_grid: ImageGrid):
    shape = (image_grid.image_height, image_grid.image_width)
    mat = np.zeros(shape, dtype=np.uint8)
    return Mask(mat=mat)


def blend_src_to_dst_image(
    src_image: Image,
    src_image_grid: ImageGrid,
    dst_image_grid: ImageGrid,
):
    map_y, map_x = src_image_grid.generate_remap_params(dst_image_grid)
    dst_image_mat = cv.remap(src_image.mat, map_x, map_y, cv.INTER_LINEAR)
    return Image(mat=dst_image_mat, mode=src_image.mode)


def blend_src_to_dst_score_map(
    src_score_map: ScoreMap,
    src_image_grid: ImageGrid,
    dst_image_grid: ImageGrid,
):
    map_y, map_x = src_image_grid.generate_remap_params(dst_image_grid)
    dst_score_map_mat = cv.remap(src_score_map.mat, map_x, map_y, cv.INTER_LINEAR)
    return ScoreMap(mat=dst_score_map_mat)


def blend_src_to_dst_mask(
    src_mask: Mask,
    src_image_grid: ImageGrid,
    dst_image_grid: ImageGrid,
):
    map_y, map_x = src_image_grid.generate_remap_params(dst_image_grid)
    dst_mask_mat = cv.remap(src_mask.mat, map_x, map_y, cv.INTER_LINEAR)
    return Mask(mat=dst_mask_mat)
