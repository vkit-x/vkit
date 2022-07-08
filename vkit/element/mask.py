from typing import cast, Optional, Tuple, Union, List, Iterable, TypeVar

import attrs
import numpy as np
import cv2 as cv

from .type import Shapable, FillByElementsMode
from .opt import generate_resized_shape, fill_np_array


@attrs.define
class MaskSetItemConfig:
    value: Union['Mask', np.ndarray, int] = 1
    keep_max_value: bool = False
    keep_min_value: bool = False


_E = TypeVar('_E', 'Box', 'Polygon')


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

    @staticmethod
    def unpack_element_value_pairs(
        element_value_pairs: Iterable[Tuple[_E, Union['Mask', np.ndarray, int]]],
    ):
        elements: List[_E] = []
        values: List[Union[Mask, np.ndarray, int]] = []
        for element, value in element_value_pairs:
            elements.append(element)
            values.append(value)
        return elements, values

    def fill_by_box_value_pairs(
        self,
        box_value_pairs: Iterable[Tuple['Box', Union['Mask', np.ndarray, int]]],
        mode: FillByElementsMode = FillByElementsMode.UNION,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
        skip_values_uniqueness_check: bool = False,
    ):
        boxes, values = self.unpack_element_value_pairs(box_value_pairs)

        boxes_mask = generate_fill_by_boxes_mask(self.shape, boxes, mode)
        if boxes_mask is None:
            for box, value in zip(boxes, values):
                box.fill_mask(
                    mask=self,
                    value=value,
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = check_elements_uniqueness(values)

            if unique:
                boxes_mask.fill_mask(
                    mask=self,
                    value=values[0],
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )
            else:
                for box, value in zip(boxes, values):
                    box_mask = box.extract_mask(boxes_mask).to_box_attached(box)
                    box_mask.fill_mask(
                        mask=self,
                        value=value,
                        keep_max_value=keep_max_value,
                        keep_min_value=keep_min_value,
                    )

    def fill_by_boxes(
        self,
        boxes: Iterable['Box'],
        value: Union['Mask', np.ndarray, int] = 1,
        mode: FillByElementsMode = FillByElementsMode.UNION,
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
        polygon_value_pairs: Iterable[Tuple['Polygon', Union['Mask', np.ndarray, int]]],
        mode: FillByElementsMode = FillByElementsMode.UNION,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
        skip_values_uniqueness_check: bool = False,
    ):
        polygons, values = self.unpack_element_value_pairs(polygon_value_pairs)

        polygons_mask = generate_fill_by_polygons_mask(self.shape, polygons, mode)
        if polygons_mask is None:
            for polygon, value in zip(polygons, values):
                polygon.fill_mask(
                    mask=self,
                    value=value,
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = check_elements_uniqueness(values)

            if unique:
                polygons_mask.fill_mask(
                    mask=self,
                    value=values[0],
                    keep_max_value=keep_max_value,
                    keep_min_value=keep_min_value,
                )
            else:
                for polygon, value in zip(polygons, values):
                    bounding_box = polygon.to_bounding_box()
                    polygon_mask = bounding_box.extract_mask(polygons_mask)
                    polygon_mask = polygon_mask.to_box_attached(bounding_box)
                    polygon_mask.fill_mask(
                        mask=self,
                        value=value,
                        keep_max_value=keep_max_value,
                        keep_min_value=keep_min_value,
                    )

    def fill_by_polygons(
        self,
        polygons: Iterable['Polygon'],
        value: Union['Mask', np.ndarray, int] = 1,
        mode: FillByElementsMode = FillByElementsMode.UNION,
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

    def __setitem__(
        self,
        element: Union['Box', 'Polygon'],
        config: Union[
            'Mask',
            np.ndarray,
            int,
            MaskSetItemConfig,
        ],
    ):  # yapf: disable
        if not isinstance(config, MaskSetItemConfig):
            value = config
            keep_max_value = False
            keep_min_value = False
        else:
            assert isinstance(config, MaskSetItemConfig)
            value = config.value
            keep_max_value = config.keep_max_value
            keep_min_value = config.keep_min_value

        element.fill_mask(
            mask=self,
            value=value,
            keep_min_value=keep_min_value,
            keep_max_value=keep_max_value,
        )

    def __getitem__(
        self,
        element: Union['Box', 'Polygon'],
    ):
        return element.extract_mask(self)

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
        resized_mask = resized_mask.to_box_attached(resized_box)
        return resized_mask

    def to_box_attached(self, box: 'Box'):
        return attrs.evolve(self, box=box)

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        alpha: Union[np.ndarray, float] = 1.0,
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

    def fill_mask(
        self,
        mask: 'Mask',
        value: Union['Mask', np.ndarray, int] = 1,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        if isinstance(value, Mask):
            value = value.mat

        self.fill_np_array(
            mask.mat,
            value,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def extract_score_map(self, score_map: 'ScoreMap'):
        if self.box:
            score_map = self.box.extract_score_map(score_map)

        score_map = score_map.copy()
        self.to_inverted_mask().fill_score_map(score_map, value=0.0)
        return score_map

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

    def extract_image(self, image: 'Image'):
        if self.box:
            image = self.box.extract_image(image)

        image = image.copy()
        self.to_inverted_mask().fill_image(image, value=0)
        return image

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        alpha: Union['ScoreMap', np.ndarray, float] = 1.0,
    ):
        if isinstance(value, Image):
            value = value.mat
        if isinstance(alpha, ScoreMap):
            assert alpha.is_prob
            alpha = alpha.mat

        self.fill_np_array(image.mat, value, alpha=alpha)


def generate_fill_by_masks_mask(
    shape: Tuple[int, int],
    masks: Iterable[Mask],
    mode: FillByElementsMode,
):
    if mode == FillByElementsMode.UNION:
        return None

    masks_mask = Mask.from_shape(shape)

    for mask in masks:
        if mask.box:
            boxed_mat = mask.box.extract_np_array(masks_mask.mat)
        else:
            boxed_mat = masks_mask.mat

        np_non_oob_mask = (boxed_mat < 255)
        boxed_mat[mask.np_mask & np_non_oob_mask] += 1

    if mode == FillByElementsMode.DISTINCT:
        masks_mask.mat[masks_mask.mat > 1] = 0

    elif mode == FillByElementsMode.INTERSECT:
        masks_mask.mat[masks_mask.mat == 1] = 0

    else:
        raise NotImplementedError()

    return masks_mask


# Cyclic dependency, by design.
from .uniqueness import check_elements_uniqueness  # noqa: E402
from .image import Image  # noqa: E402
from .box import Box, generate_fill_by_boxes_mask  # noqa: E402
from .polygon import Polygon, generate_fill_by_polygons_mask  # noqa: E402
from .score_map import ScoreMap  # noqa: E402
