import numpy as np
import cv2 as cv

from vkit.element import (
    Image,
    ImageKind,
    ScoreMap,
    Mask,
)
from .type import ImageGrid


def create_image_from_image_grid(image_grid: ImageGrid, image_kind: ImageKind):
    ndim = ImageKind.to_ndim(image_kind)
    if ndim == 2:
        shape = (image_grid.image_height, image_grid.image_width)
    elif ndim == 3:
        num_channels = ImageKind.to_num_channels(image_kind)
        assert num_channels
        shape = (image_grid.image_height, image_grid.image_width, num_channels)
    else:
        raise NotImplementedError()

    dtype = ImageKind.to_dtype(image_kind)
    mat = np.zeros(shape, dtype=dtype)
    return Image(mat=mat, kind=image_kind)


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
    return Image(mat=dst_image_mat, kind=src_image.kind)


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
