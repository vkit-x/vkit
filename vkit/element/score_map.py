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
from typing import Optional, Tuple, Union, Iterable, Callable, List, TypeVar
from contextlib import ContextDecorator

import attrs
import numpy as np
import cv2 as cv

from .type import Shapable, ElementSetOperationMode
from .opt import generate_shape_and_resized_shape


@attrs.define
class ScoreMapSetItemConfig:
    value: Union[
        'ScoreMap',
        np.ndarray,
        float,
    ] = 1.0  # yapf: disable
    keep_max_value: bool = False
    keep_min_value: bool = False


@attrs.define
class NpVec:
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def from_point(cls, point: 'Point'):
        return cls(
            x=np.asarray(point.smooth_x, dtype=np.float32),
            y=np.asarray(point.smooth_y, dtype=np.float32),
        )

    def __add__(self, other: 'NpVec'):
        return NpVec(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: 'NpVec'):
        return NpVec(x=self.x - other.x, y=self.y - other.y)  # type: ignore

    def __mul__(self, other: 'NpVec') -> np.ndarray:
        return self.x * other.y - self.y * other.x  # type: ignore


class WritableScoreMapContextDecorator(ContextDecorator):

    def __init__(self, score_map: 'ScoreMap'):
        super().__init__()
        self.score_map = score_map

    def __enter__(self):
        if self.score_map.mat.flags.c_contiguous:
            assert not self.score_map.mat.flags.writeable

        try:
            self.score_map.mat.flags.writeable = True
        except ValueError:
            # Copy on write.
            object.__setattr__(
                self.score_map,
                'mat',
                np.array(self.score_map.mat),
            )
            assert self.score_map.mat.flags.writeable

    def __exit__(self, *exc):  # type: ignore
        self.score_map.mat.flags.writeable = False


_E = TypeVar('_E', 'Box', 'Polygon', 'Mask')


@attrs.define(frozen=True, eq=False)
class ScoreMap(Shapable):
    mat: np.ndarray
    box: Optional['Box'] = None
    is_prob: bool = True

    def __attrs_post_init__(self):
        if self.mat.ndim != 2:
            raise RuntimeError('ndim should == 2.')
        if self.box and self.shape != self.box.shape:
            raise RuntimeError('self.shape != box.shape.')

        if self.mat.dtype != np.float32:
            raise RuntimeError('mat.dtype != np.float32')

        # For the control of write.
        self.mat.flags.writeable = False

        if self.is_prob:
            score_min = self.mat.min()
            score_max = self.mat.max()
            if score_min < 0.0 or score_max > 1.0:
                raise RuntimeError('score not in range [0.0, 1.0]')

    ###############
    # Constructor #
    ###############
    @classmethod
    def from_shape(
        cls,
        shape: Tuple[int, int],
        value: float = 0.0,
        is_prob: bool = True,
    ):
        height, width = shape
        if is_prob:
            assert 0.0 <= value <= 1.0
        mat = np.full((height, width), fill_value=value, dtype=np.float32)
        return cls(mat=mat, is_prob=is_prob)

    @classmethod
    def from_shapable(
        cls,
        shapable: Shapable,
        value: float = 0.0,
        is_prob: bool = True,
    ):
        return cls.from_shape(
            shape=shapable.shape,
            value=value,
            is_prob=is_prob,
        )

    @classmethod
    def from_quad_interpolation(
        cls,
        point0: 'Point',
        point1: 'Point',
        point2: 'Point',
        point3: 'Point',
        func_np_uv_to_mat: Callable[[np.ndarray], np.ndarray],
        is_prob: bool = True,
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
        u in [0.0, 1.0], and
        u -> 0.0, x -> line (point0, point3)
        u -> 1.0, x -> line (point1, point2)

        v in [0.0, 1.0], and
        v -> 0.0, x -> line (point0, point1)
        v -> 1.0, x -> line (point3, point2)
        '''
        polygon = Polygon.create((
            point0,
            point1,
            point2,
            point3,
        ))
        bounding_box = polygon.bounding_box
        self_relative_polygon = polygon.self_relative_polygon
        np_active_mask = polygon.internals.np_mask

        np_vec_0 = NpVec.from_point(self_relative_polygon.points[0])
        np_vec_1 = NpVec.from_point(self_relative_polygon.points[1])
        np_vec_2 = NpVec.from_point(self_relative_polygon.points[2])
        np_vec_3 = NpVec.from_point(self_relative_polygon.points[3])

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

            np_masked_v_pos: np.ndarray = np_v_pos[np_active_mask]
            np_v_pos_valid = (0.0 <= np_masked_v_pos) & (np_masked_v_pos <= 1.0)

            np_masked_v_neg: np.ndarray = np_v_neg[np_active_mask]
            np_v_neg_valid = (0.0 <= np_masked_v_neg) & (np_masked_v_neg <= 1.0)

            if np_v_pos_valid.sum() >= np_v_neg_valid.sum():
                np_v = np_v_pos
            else:
                np_v = np_v_neg

        np_v[~np_active_mask] = 0.0
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

        np_u[~np_active_mask] = 0.0
        np_u = np.clip(np_u, 0.0, 1.0)

        # Stack to (height, width, 2)
        np_uv = np.stack((np_u, np_v), axis=-1)

        # Mat.
        mat = func_np_uv_to_mat(np_uv)
        return cls(
            mat=mat,
            box=bounding_box,
            is_prob=is_prob,
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

    @property
    def equivalent_box(self):
        return self.box or Box.from_shapable(self)

    @property
    def writable_context(self):
        return WritableScoreMapContextDecorator(self)

    ############
    # Operator #
    ############
    def copy(self):
        return attrs.evolve(self, mat=self.mat.copy())

    def assign_mat(self, mat: np.ndarray):
        with self.writable_context:
            object.__setattr__(self, 'mat', mat)

    @classmethod
    def unpack_element_value_pairs(
        cls,
        is_prob: bool,
        element_value_pairs: Iterable[Tuple[_E, Union['ScoreMap', np.ndarray, float]]],
    ):
        elements: List[_E] = []

        values: List[Union[ScoreMap, np.ndarray, float]] = []
        for box, value in element_value_pairs:
            elements.append(box)

            if is_prob:
                if isinstance(value, ScoreMap):
                    assert (0.0 <= value.mat).all() and (value.mat <= 1.0).all()
                elif isinstance(value, np.ndarray):
                    assert (0.0 <= value).all() and (value <= 1.0).all()
                elif isinstance(value, float):
                    assert 0.0 <= value <= 1.0
                else:
                    raise NotImplementedError()
            values.append(value)

        return elements, values

    def fill_by_box_value_pairs(
        self,
        box_value_pairs: Iterable[Tuple['Box', Union['ScoreMap', np.ndarray, float]]],
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
        skip_values_uniqueness_check: bool = False,
    ):
        boxes, values = self.unpack_element_value_pairs(self.is_prob, box_value_pairs)

        boxes_mask = generate_fill_by_boxes_mask(self.shape, boxes, mode)
        if boxes_mask is None:
            for box, value in zip(boxes, values):
                box.fill_score_map(
                    score_map=self,
                    value=value,
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = check_elements_uniqueness(values)

            if unique:
                boxes_mask.fill_score_map(
                    score_map=self,
                    value=values[0],
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )
            else:
                for box, value in zip(boxes, values):
                    box_mask = box.extract_mask(boxes_mask).to_box_attached(box)
                    box_mask.fill_score_map(
                        score_map=self,
                        value=value,
                        keep_max_value=keep_max_value,
                        keep_min_value=keep_min_value,
                    )

    def fill_by_boxes(
        self,
        boxes: Iterable['Box'],
        value: Union['ScoreMap', np.ndarray, float] = 1.0,
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        self.fill_by_box_value_pairs(
            box_value_pairs=((box, value) for box in boxes),
            mode=mode,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
            skip_values_uniqueness_check=True,
        )

    def fill_by_polygon_value_pairs(
        self,
        polygon_value_pairs: Iterable[Tuple['Polygon', Union['ScoreMap', np.ndarray, float]]],
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
        skip_values_uniqueness_check: bool = False,
    ):
        polygons, values = self.unpack_element_value_pairs(self.is_prob, polygon_value_pairs)

        polygons_mask = generate_fill_by_polygons_mask(self.shape, polygons, mode)
        if polygons_mask is None:
            for polygon, value in zip(polygons, values):
                polygon.fill_score_map(
                    score_map=self,
                    value=value,
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = check_elements_uniqueness(values)

            if unique:
                polygons_mask.fill_score_map(
                    score_map=self,
                    value=values[0],
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )
            else:
                for polygon, value in zip(polygons, values):
                    bounding_box = polygon.to_bounding_box()
                    polygon_mask = bounding_box.extract_mask(polygons_mask)
                    polygon_mask = polygon_mask.to_box_attached(bounding_box)
                    polygon_mask.fill_score_map(
                        score_map=self,
                        value=value,
                        keep_max_value=keep_max_value,
                        keep_min_value=keep_min_value,
                    )

    def fill_by_polygons(
        self,
        polygons: Iterable['Polygon'],
        value: Union['ScoreMap', np.ndarray, float] = 1.0,
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        self.fill_by_polygon_value_pairs(
            polygon_value_pairs=((polygon, value) for polygon in polygons),
            mode=mode,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
            skip_values_uniqueness_check=True,
        )

    def fill_by_mask_value_pairs(
        self,
        mask_value_pairs: Iterable[Tuple['Mask', Union['ScoreMap', np.ndarray, float]]],
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
        skip_values_uniqueness_check: bool = False,
    ):
        masks, values = self.unpack_element_value_pairs(self.is_prob, mask_value_pairs)

        masks_mask = generate_fill_by_masks_mask(self.shape, masks, mode)
        if masks_mask is None:
            for mask, value in zip(masks, values):
                mask.fill_score_map(
                    score_map=self,
                    value=value,
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = check_elements_uniqueness(values)

            if unique:
                masks_mask.fill_score_map(
                    score_map=self,
                    value=values[0],
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )
            else:
                for mask, value in zip(masks, values):
                    if mask.box:
                        boxed_mask = mask.box.extract_mask(masks_mask)
                    else:
                        boxed_mask = masks_mask

                    boxed_mask = boxed_mask.copy()
                    mask.to_inverted_mask().fill_mask(boxed_mask, value=0)
                    boxed_mask.fill_score_map(
                        score_map=self,
                        value=value,
                        keep_max_value=keep_max_value,
                        keep_min_value=keep_min_value,
                    )

    def fill_by_masks(
        self,
        masks: Iterable['Mask'],
        value: Union['ScoreMap', np.ndarray, float] = 1.0,
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        self.fill_by_mask_value_pairs(
            mask_value_pairs=((mask, value) for mask in masks),
            mode=mode,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
            skip_values_uniqueness_check=True,
        )

    def __setitem__(
        self,
        element: Union[
            'Box',
            'Polygon',
            'Mask',
        ],
        config: Union[
            'ScoreMap',
            np.ndarray,
            float,
            ScoreMapSetItemConfig,
        ],
    ):  # yapf: disable
        if not isinstance(config, ScoreMapSetItemConfig):
            value = config
            keep_max_value = False
            keep_min_value = False
        else:
            assert isinstance(config, ScoreMapSetItemConfig)
            value = config.value
            keep_max_value = config.keep_max_value
            keep_min_value = config.keep_min_value

        element.fill_score_map(
            score_map=self,
            value=value,
            keep_min_value=keep_min_value,
            keep_max_value=keep_max_value,
        )

    def __getitem__(
        self,
        element: Union[
            'Box',
            'Polygon',
            'Mask',
        ],
    ):  # yapf: disable
        return element.extract_score_map(self)

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
        score_map = self.from_quad_interpolation(
            point0=point0,
            point1=point1,
            point2=point2,
            point3=point3,
            func_np_uv_to_mat=func_np_uv_to_mat,
            is_prob=self.is_prob,
        )
        assert score_map.box
        with self.writable_context:
            score_map.box.fill_np_array(
                mat=self.mat,
                value=score_map.mat,
                np_mask=(score_map.mat > 0.0),
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def to_shifted_score_map(self, offset_y: int = 0, offset_x: int = 0):
        assert self.box
        shifted_box = self.box.to_shifted_box(offset_y=offset_y, offset_x=offset_x)
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
        resized_score_map = self.to_box_detached().to_resized_score_map(
            resized_height=resized_box.height,
            resized_width=resized_box.width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_score_map = resized_score_map.to_box_attached(resized_box)
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
        if self.is_prob:
            # NOTE: Interpolation like bi-cubic could generate out-of-bound values.
            mat = np.clip(mat, 0.0, 1.0)
        return attrs.evolve(self, mat=mat)

    def to_cropped_score_map(
        self,
        up: Optional[int] = None,
        down: Optional[int] = None,
        left: Optional[int] = None,
        right: Optional[int] = None,
    ):
        assert not self.box

        up = up or 0
        down = down or self.height - 1
        left = left or 0
        right = right or self.width - 1

        return attrs.evolve(self, mat=self.mat[up:down + 1, left:right + 1])

    def to_box_attached(self, box: 'Box'):
        assert self.height == box.height
        assert self.width == box.width
        return attrs.evolve(self, box=box)

    def to_box_detached(self):
        assert self.box
        return attrs.evolve(self, box=None)

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        self.equivalent_box.fill_np_array(
            mat=mat,
            value=value,
            alpha=self,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
    ):
        self.equivalent_box.fill_image(
            image=image,
            value=value,
            alpha=self,
        )

    def to_mask(self, threshold: float = 0.0):
        mat = (self.mat > threshold).astype(np.uint8)
        return Mask(mat=mat, box=self.box)


def generate_fill_by_score_maps_mask(
    shape: Tuple[int, int],
    score_maps: Iterable[ScoreMap],
    mode: ElementSetOperationMode,
):
    if mode == ElementSetOperationMode.UNION:
        return None
    else:
        return Mask.from_score_maps(shape, score_maps, mode)


# Cyclic dependency, by design.
from .uniqueness import check_elements_uniqueness  # noqa: E402
from .image import Image  # noqa: E402
from .point import Point  # noqa: E402
from .box import Box, generate_fill_by_boxes_mask  # noqa: E402
from .mask import Mask, generate_fill_by_masks_mask  # noqa: E402
from .polygon import Polygon, generate_fill_by_polygons_mask  # noqa: E402
