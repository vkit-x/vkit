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
from typing import (
    Generic,
    Type,
    TypeVar,
    Tuple,
    Optional,
)

import numpy as np
from numpy.random import Generator as RandomGenerator

from vkit.element import (
    Image,
    Point,
    Mask,
    ScoreMap,
)
from ...interface import (
    DistortionConfig,
    DistortionState,
    Distortion,
)
from .type import ImageGrid
from .point_projector import PointProjector
from .grid_creator import create_dst_image_grid_and_shift_amounts_and_resize_ratios
from .grid_blender import (
    blend_src_to_dst_image,
    blend_src_to_dst_score_map,
    blend_src_to_dst_mask,
)

_T_CONFIG = TypeVar('_T_CONFIG', bound=DistortionConfig)


class DistortionStateImageGridBased(DistortionState[_T_CONFIG]):

    @property
    def shift_amount_y(self):
        return self._shift_amount_y

    @shift_amount_y.setter
    def shift_amount_y(self, val: float):
        self._shift_amount_y = val

    @property
    def shift_amount_x(self):
        return self._shift_amount_x

    @shift_amount_x.setter
    def shift_amount_x(self, val: float):
        self._shift_amount_x = val

    @property
    def resize_ratio_y(self):
        return self._resize_ratio_y

    @resize_ratio_y.setter
    def resize_ratio_y(self, val: float):
        self._resize_ratio_y = val

    @property
    def resize_ratio_x(self):
        return self._resize_ratio_x

    @resize_ratio_x.setter
    def resize_ratio_x(self, val: float):
        self._resize_ratio_x = val

    @property
    def src_image_grid(self):
        return self._src_image_grid

    @src_image_grid.setter
    def src_image_grid(self, val: ImageGrid):
        self._src_image_grid = val

    @property
    def dst_image_grid(self):
        return self._dst_image_grid

    @dst_image_grid.setter
    def dst_image_grid(self, val: ImageGrid):
        self._dst_image_grid = val

    def initialize_image_grid_based(
        self,
        src_image_grid: ImageGrid,
        point_projector: PointProjector,
        resize_as_src: bool = False,
    ):
        self.src_image_grid = src_image_grid

        (
            self.dst_image_grid,
            (self.shift_amount_y, self.shift_amount_x),
            (self.resize_ratio_y, self.resize_ratio_x),
        ) = create_dst_image_grid_and_shift_amounts_and_resize_ratios(
            self.src_image_grid,
            point_projector,
            resize_as_src=resize_as_src,
        )

    def shift_and_resize_point(self, point: Point):
        return Point.create(
            y=(point.smooth_y - self.shift_amount_y) * self.resize_ratio_y,
            x=(point.smooth_x - self.shift_amount_x) * self.resize_ratio_x,
        )

    @property
    def result_shape(self):
        return (self.dst_image_grid.image_height, self.dst_image_grid.image_width)


_T_STATE = TypeVar('_T_STATE', bound=DistortionStateImageGridBased)


class FuncImageGridBased(Generic[_T_CONFIG, _T_STATE]):

    @classmethod
    def func_image(
        cls,
        config: _T_CONFIG,
        state: Optional[_T_STATE],
        image: Image,
        rng: Optional[RandomGenerator],
    ):
        assert state
        return blend_src_to_dst_image(
            image,
            state.src_image_grid,
            state.dst_image_grid,
        )

    @classmethod
    def func_score_map(
        cls,
        config: _T_CONFIG,
        state: Optional[_T_STATE],
        score_map: ScoreMap,
        rng: Optional[RandomGenerator],
    ):
        assert state
        return blend_src_to_dst_score_map(
            score_map,
            state.src_image_grid,
            state.dst_image_grid,
        )

    @classmethod
    def func_mask(
        cls,
        config: _T_CONFIG,
        state: Optional[_T_STATE],
        mask: Mask,
        rng: Optional[RandomGenerator],
    ):
        assert state
        return blend_src_to_dst_mask(
            mask,
            state.src_image_grid,
            state.dst_image_grid,
        )

    @classmethod
    def func_active_mask(
        cls,
        config: _T_CONFIG,
        state: Optional[_T_STATE],
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        assert state
        border_polygon = state.dst_image_grid.generate_border_polygon()
        active_mask = Mask.from_shape((
            state.dst_image_grid.image_height,
            state.dst_image_grid.image_width,
        ))
        border_polygon.fill_mask(active_mask)
        return active_mask

    @classmethod
    def func_point(
        cls,
        config: _T_CONFIG,
        state: Optional[_T_STATE],
        shape: Tuple[int, int],
        point: Point,
        rng: Optional[RandomGenerator],
    ):
        assert state
        src_image_grid = state.src_image_grid
        dst_image_grid = state.dst_image_grid

        assert src_image_grid.grid_size
        polygon_row = point.y // src_image_grid.grid_size
        polygon_col = point.x // src_image_grid.grid_size

        trans_mat = src_image_grid.get_trans_mat(polygon_row, polygon_col, dst_image_grid)
        dst_tx, dst_ty, dst_t = np.matmul(trans_mat, (point.smooth_x, point.smooth_y, 1.0))
        return Point.create(
            y=float(dst_ty / dst_t),
            x=float(dst_tx / dst_t),
        )


class DistortionImageGridBased(Distortion[_T_CONFIG, _T_STATE]):

    def __init__(
        self,
        config_cls: Type[_T_CONFIG],
        state_cls: Type[_T_STATE],
    ):
        func_cls = FuncImageGridBased[_T_CONFIG, _T_STATE]
        super().__init__(
            config_cls=config_cls,
            state_cls=state_cls,
            func_image=func_cls.func_image,
            func_mask=func_cls.func_mask,
            func_score_map=func_cls.func_score_map,
            func_active_mask=func_cls.func_active_mask,
            func_point=func_cls.func_point,
        )
