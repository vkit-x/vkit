from typing import cast, Optional, Tuple, Union, Iterable
from enum import Enum, unique

import attrs
import numpy as np
import cv2 as cv

from .type import Shapable
from .opt import (
    generate_resized_shape,
    fill_np_array,
)


@unique
class MaskFillByPolygonsMode(Enum):
    UNION = 'union'
    DISTINCT = 'distinct'
    INTERSECTION = 'intersection'


@attrs.define
class Mask(Shapable):
    mat: np.ndarray
    box: Optional['Box'] = None

    def __attrs_post_init__(self):
        if self.mat.dtype != np.uint8:
            raise RuntimeError('mat.dtype != np.uint8')
        if self.mat.ndim != 2:
            raise RuntimeError('ndim should == 2.')
        if self.box and self.shape != self.box.shape:
            raise RuntimeError('self.shape != box.shape.')

    ###############
    # Constructor #
    ###############
    @staticmethod
    def from_shape(shape: Tuple[int, int], value: int = 0):
        height, width = shape
        if value == 0:
            np_init_func = np.zeros
        else:
            assert value == 1
            np_init_func = np.ones
        mat = np_init_func((height, width), dtype=np.uint8)
        return Mask(mat=mat)

    @staticmethod
    def from_shapable(shapable: Shapable, value: int = 0):
        return Mask.from_shape(shape=shapable.shape, value=value)

    ############
    # Property #
    ############
    @property
    def height(self):
        return self.mat.shape[0]

    @property
    def width(self):
        return self.mat.shape[1]

    @property
    def np_mask(self):
        return (self.mat > 0)

    ############
    # Operator #
    ############
    def copy(self):
        return attrs.evolve(self, mat=self.mat.copy())

    def fill_by_polygons(
        self,
        polygons: Iterable['Polygon'],
        mode: MaskFillByPolygonsMode = MaskFillByPolygonsMode.UNION,
    ):
        if mode == MaskFillByPolygonsMode.UNION:
            for polygon in polygons:
                polygon.fill_mask(self)

        else:
            for polygon in polygons:
                internals = polygon.to_fill_np_array_internals()
                boxed_mat = internals.bounding_box.extract_np_array(self.mat)
                np_polygon_mask = internals.get_np_mask()
                np_non_oob_mask = (boxed_mat < 255)
                boxed_mat[np_polygon_mask & np_non_oob_mask] += 1

            if mode == MaskFillByPolygonsMode.DISTINCT:
                self.mat[self.mat > 1] = 0

            elif mode == MaskFillByPolygonsMode.INTERSECTION:
                self.mat[self.mat == 1] = 0

            else:
                raise NotImplementedError()

    def to_inverted_mask(self):
        mat = (~self.np_mask).astype(np.uint8)
        return attrs.evolve(self, mat=mat)

    def to_shifted_mask(self, y_offset: int = 0, x_offset: int = 0):
        assert self.box
        shifted_box = self.box.to_shifted_box(y_offset=y_offset, x_offset=x_offset)
        return attrs.evolve(self, box=shifted_box)

    def to_resized_mask(
        self,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
        cv_resize_interpolation: int = cv.INTER_CUBIC,
        binarization_threshold: int = 0,
    ):
        assert not self.box
        resized_height, resized_width = generate_resized_shape(
            height=self.height,
            width=self.width,
            resized_height=resized_height,
            resized_width=resized_width,
        )

        # Deal with precision loss.
        mat = self.np_mask.astype(np.uint8) * 255
        mat = cv.resize(
            mat,
            (resized_width, resized_height),
            interpolation=cv_resize_interpolation,
        )
        mat = cast(np.ndarray, mat)
        mat = (mat > binarization_threshold).astype(np.uint8)

        return Mask(mat=mat)

    def to_conducted_resized_mask(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
        cv_resize_interpolation: int = cv.INTER_CUBIC,
        binarization_threshold: int = 0,
    ):
        assert self.box
        resized_box = self.box.to_conducted_resized_box(
            shapable_or_shape=shapable_or_shape,
            resized_height=resized_height,
            resized_width=resized_width,
        )
        resized_mask = self.to_resized_mask(
            resized_height=resized_box.height,
            resized_width=resized_box.width,
            cv_resize_interpolation=cv_resize_interpolation,
            binarization_threshold=binarization_threshold,
        )
        resized_mask.box = resized_box
        return resized_mask

    def to_box_attached(self, box: 'Box'):
        return attrs.evolve(self, box=box)

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        alpha: Union[float, np.ndarray] = 1.0,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        if self.box:
            self.box.fill_np_array(
                mat=mat,
                value=value,
                np_mask=self.np_mask,
                alpha=alpha,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

        else:
            fill_np_array(
                mat=mat,
                value=value,
                np_mask=self.np_mask,
                alpha=alpha,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        alpha: Union[float, np.ndarray] = 1.0,
    ):
        if isinstance(value, Image):
            value = value.mat

        self.fill_np_array(image.mat, value, alpha=alpha)

    def fill_mask(
        self,
        mask: 'Mask',
        value: Union['Mask', np.ndarray, int] = 1,
    ):
        if isinstance(value, Mask):
            value = value.mat

        self.fill_np_array(mask.mat, value)

    def fill_score_map(
        self,
        score_map: 'ScoreMap',
        value: Union['ScoreMap', np.ndarray, float],
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        if isinstance(value, ScoreMap):
            value = value.mat

        self.fill_np_array(
            score_map.mat,
            value,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def to_score_map(self):
        mat = self.np_mask.astype(np.float32)
        return ScoreMap(mat=mat, box=self.box)


# Cyclic dependency, by design.
from .image import Image  # noqa: E402
from .box import Box  # noqa: E402
from .score_map import ScoreMap  # noqa: E402
from .polygon import Polygon  # noqa: E402
