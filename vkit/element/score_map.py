from typing import Optional, Tuple, Union, Iterable, Callable

import attrs
import numpy as np
import cv2 as cv

from .type import Shapable
from .opt import (
    generate_shape_and_resized_shape,
    fill_np_array,
)


@attrs.define
class NpVec:
    x: np.ndarray
    y: np.ndarray

    @staticmethod
    def from_point(point: 'Point'):
        return NpVec(
            x=np.asarray(point.x, dtype=np.float32),
            y=np.asarray(point.y, dtype=np.float32),
        )

    def __add__(self, other: 'NpVec'):
        return NpVec(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: 'NpVec'):
        return NpVec(x=self.x - other.x, y=self.y - other.y)  # type: ignore

    def __mul__(self, other: 'NpVec') -> np.ndarray:
        return self.x * other.y - self.y * other.x  # type: ignore


@attrs.define
class ScoreMap(Shapable):
    mat: np.ndarray
    box: Optional['Box'] = None
    score_as_prob: bool = True

    def __attrs_post_init__(self):
        if self.mat.ndim != 2:
            raise RuntimeError('ndim should == 2.')
        if self.box and self.shape != self.box.shape:
            raise RuntimeError('self.shape != box.shape.')

        if self.mat.dtype != np.float32:
            raise RuntimeError('mat.dtype != np.float32')

        if self.score_as_prob:
            score_min = self.mat.min()
            score_max = self.mat.max()
            if score_min < 0.0 or score_max > 1.0:
                raise RuntimeError('score not in range [0.0, 1.0]')

    ###############
    # Constructor #
    ###############
    @staticmethod
    def from_shape(
        shape: Tuple[int, int],
        value: float = 0.0,
        score_as_prob: bool = True,
    ):
        height, width = shape
        if score_as_prob:
            assert 0.0 <= value <= 1.0
        mat = np.full((height, width), fill_value=value, dtype=np.float32)
        return ScoreMap(mat=mat, score_as_prob=score_as_prob)

    @staticmethod
    def from_shapable(
        shapable: Shapable,
        value: float = 0.0,
        score_as_prob: bool = True,
    ):
        return ScoreMap.from_shape(
            shape=shapable.shape,
            value=value,
            score_as_prob=score_as_prob,
        )

    @staticmethod
    def from_quad_interpolation(
        point0: 'Point',
        point1: 'Point',
        point2: 'Point',
        point3: 'Point',
        func_np_uv_to_mat: Callable[[np.ndarray], np.ndarray],
        score_as_prob: bool = True,
    ):
        '''
        Ref: https://www.reedbeta.com/blog/quadrilateral-interpolation-part-2/

        points in clockwise order:
        point0 0.0 -- u --> 1.0 point1
                                 0.0
                                  ↓
                                  v
                                  ↓
                                 1.0
        point3 0.0 <-- u -- 1.0 point2

        lerp(a, b, r) = a + r * (b - a)
        pointx(u, v) = lerp(
            lerp(point0, point1, u),
            lerp(point3, point2, u),
            v,
        )

        Hence,
        u in [0.0, 1.0]
        u -> 0.0, x -> line (point0, point3)
        x -> 1.0, x -> line (point1, point2)

        v in [0.0, 1.0]
        v -> 0.0, x -> line (point0, point1)
        v -> 1.0, x -> line (point3, point2)
        '''
        polygon = Polygon.create((
            point0,
            point1,
            point2,
            point3,
        ))
        internals = polygon.to_fill_np_array_internals()
        bounding_box = internals.bounding_box
        shifted_polygon = internals.get_shifted_polygon()
        np_mask = internals.get_np_mask()

        np_vec_0 = NpVec.from_point(shifted_polygon.points[0])
        np_vec_1 = NpVec.from_point(shifted_polygon.points[1])
        np_vec_2 = NpVec.from_point(shifted_polygon.points[2])
        np_vec_3 = NpVec.from_point(shifted_polygon.points[3])

        # F(x, y) -> x
        np_pointx_x = np.repeat(
            np.expand_dims(
                np.arange(bounding_box.width, dtype=np.int32),
                axis=0,
            ),
            bounding_box.height,
            axis=0,
        )
        # F(x, y) -> y
        np_pointx_y = np.repeat(
            np.expand_dims(
                np.arange(bounding_box.height, dtype=np.int32),
                axis=1,
            ),
            bounding_box.width,
            axis=1,
        )
        np_vec_x = NpVec(x=np_pointx_x, y=np_pointx_y)

        np_vec_q = np_vec_x - np_vec_0
        np_vec_b1 = np_vec_1 - np_vec_0
        np_vec_b2 = np_vec_3 - np_vec_0
        np_vec_b3 = ((np_vec_0 - np_vec_1) - np_vec_3) + np_vec_2

        scale_a = float(np_vec_b2 * np_vec_b3)

        np_b: np.ndarray = (np_vec_b3 * np_vec_q) - (np_vec_b1 * np_vec_b2)  # type: ignore
        np_b = np_b.astype(np.float32)

        np_c = np_vec_b1 * np_vec_q
        np_c = np_c.astype(np.float32)

        # Solve v.
        if abs(scale_a) < 0.001:
            np_v = -np_c / np_b

        else:
            np_discrim = np.sqrt(np.power(np_b, 2) - 4 * scale_a * np_c)
            scale_i2a = 0.5 / scale_a
            np_v_pos = (-np_b + np_discrim) * scale_i2a
            np_v_neg: np.ndarray = (-np_b - np_discrim) * scale_i2a  # type: ignore

            np_masked_v_pos: np.ndarray = np_v_pos[np_mask]
            np_v_pos_valid = (0.0 <= np_masked_v_pos) & (np_masked_v_pos <= 1.0)

            np_masked_v_neg: np.ndarray = np_v_neg[np_mask]
            np_v_neg_valid = (0.0 <= np_masked_v_neg) & (np_masked_v_neg <= 1.0)

            if np_v_pos_valid.sum() >= np_v_neg_valid.sum():
                np_v = np_v_pos
            else:
                np_v = np_v_neg

        np_v[~np_mask] = 0.0
        np_v = np.clip(np_v, 0.0, 1.0)

        # Solve u.
        np_u = np.zeros_like(np_v)

        np_denom_x: np.ndarray = np_vec_b1.x + np_vec_b3.x * np_v
        np_denom_y: np.ndarray = np_vec_b1.y + np_vec_b3.y * np_v

        np_denom_x_mask = (np.abs(np_denom_x) > np.abs(np_denom_y)) & (np_denom_x != 0.0)
        if np_denom_x_mask.any():
            np_q_x = np_vec_q.x
            np_u[np_denom_x_mask] = (
                (np_q_x[np_denom_x_mask] - np_vec_b2.x * np_v[np_denom_x_mask])
                / np_denom_x[np_denom_x_mask]
            )

        np_denom_y_mask = (~np_denom_x_mask) & (np_denom_y != 0.0)
        if np_denom_y_mask.any():
            np_q_y = np_vec_q.y
            np_u[np_denom_y_mask] = (
                (np_q_y[np_denom_y_mask] - np_vec_b2.y * np_v[np_denom_y_mask])
                / np_denom_y[np_denom_y_mask]
            )

        np_u[~np_mask] = 0.0
        np_u = np.clip(np_u, 0.0, 1.0)

        # Stack to (height, width, 2)
        np_uv = np.stack((np_u, np_v), axis=-1)

        # Mat.
        mat = func_np_uv_to_mat(np_uv)
        return ScoreMap(
            mat=mat,
            box=bounding_box,
            score_as_prob=score_as_prob,
        )

    ############
    # Property #
    ############
    @property
    def height(self):
        return self.mat.shape[0]

    @property
    def width(self):
        return self.mat.shape[1]

    ############
    # Operator #
    ############
    def copy(self):
        return attrs.evolve(self, mat=self.mat.copy())

    def fill_by_polygon_value_pairs(
        self,
        polygon_value_pairs: Iterable[Tuple['Polygon', float]],
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        for polygon, value in polygon_value_pairs:
            if self.score_as_prob:
                assert 0.0 <= value <= 1.0
            polygon.fill_score_map(
                self,
                value,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def fill_by_polygons(
        self,
        polygons: Iterable['Polygon'],
        value: float = 1.0,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        self.fill_by_polygon_value_pairs(
            polygon_value_pairs=((polygon, value) for polygon in polygons),
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def fill_by_quad_interpolation(
        self,
        point0: 'Point',
        point1: 'Point',
        point2: 'Point',
        point3: 'Point',
        func_np_uv_to_mat: Callable[[np.ndarray], np.ndarray],
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        score_map = ScoreMap.from_quad_interpolation(
            point0=point0,
            point1=point1,
            point2=point2,
            point3=point3,
            func_np_uv_to_mat=func_np_uv_to_mat,
            score_as_prob=self.score_as_prob,
        )
        np_non_zero_mask = (score_map.mat > 0.0)
        assert score_map.box
        score_map.box.fill_np_array(
            mat=self.mat,
            value=score_map.mat,
            np_mask=np_non_zero_mask,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def to_shifted_score_map(self, y_offset: int = 0, x_offset: int = 0):
        assert self.box
        shifted_box = self.box.to_shifted_box(y_offset=y_offset, x_offset=x_offset)
        return attrs.evolve(self, box=shifted_box)

    def to_conducted_resized_polygon(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
        cv_resize_interpolation: int = cv.INTER_CUBIC,
    ):
        assert self.box
        resized_box = self.box.to_conducted_resized_box(
            shapable_or_shape=shapable_or_shape,
            resized_height=resized_height,
            resized_width=resized_width,
        )
        resized_score_map = self.to_resized_score_map(
            resized_height=resized_box.height,
            resized_width=resized_box.width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_score_map.box = resized_box
        return resized_score_map

    def to_resized_score_map(
        self,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
        cv_resize_interpolation: int = cv.INTER_CUBIC,
    ):
        assert not self.box
        _, _, resized_height, resized_width = generate_shape_and_resized_shape(
            shapable_or_shape=self.shape,
            resized_height=resized_height,
            resized_width=resized_width,
        )
        mat = cv.resize(
            self.mat,
            (resized_width, resized_height),
            interpolation=cv_resize_interpolation,
        )
        if self.score_as_prob:
            # NOTE: Interpolation like bi-cubic could generate out-of-bound values.
            mat = np.clip(mat, 0.0, 1.0)
        return attrs.evolve(self, mat=mat)

    def to_box_attached(self, box: 'Box'):
        return attrs.evolve(self, box=box)

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        np_non_zero_mask = (self.mat > 0)

        if self.box:
            self.box.fill_np_array(
                mat=mat,
                value=value,
                np_mask=np_non_zero_mask,
                alpha=self.mat,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

        else:
            fill_np_array(
                mat=mat,
                value=value,
                np_mask=np_non_zero_mask,
                alpha=self.mat,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
    ):
        if isinstance(value, Image):
            value = value.mat

        self.fill_np_array(image.mat, value)

    def to_mask(self, threshold: float = 0.0):
        mat = (self.mat > threshold).astype(np.uint8)
        return Mask(mat=mat, box=self.box)


# Cyclic dependency, by design.
from .image import Image  # noqa: E402
from .box import Box  # noqa: E402
from .mask import Mask  # noqa: E402
from .point import Point  # noqa: E402
from .polygon import Polygon  # noqa: E402
