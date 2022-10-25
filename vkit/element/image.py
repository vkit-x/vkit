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
from typing import cast, Mapping, Sequence, Tuple, Union, Optional, Iterable, List, TypeVar
from enum import Enum, unique
from collections import abc
from contextlib import ContextDecorator

import attrs
import numpy as np
from PIL import (
    Image as PilImage,
    ImageOps as PilImageOps,
)
import cv2 as cv
import iolite as io

from vkit.utility import PathType
from .type import Shapable, ElementSetOperationMode
from .opt import generate_shape_and_resized_shape


@unique
class ImageMode(Enum):
    RGB = 'rgb'
    RGB_GCN = 'rgb_gcn'
    RGBA = 'rgba'
    HSV = 'hsv'
    HSV_GCN = 'hsv_gcn'
    HSL = 'hsl'
    HSL_GCN = 'hsl_gcn'
    GRAYSCALE = 'grayscale'
    GRAYSCALE_GCN = 'grayscale_gcn'
    NONE = 'none'

    def to_ndim(self):
        if self in _IMAGE_MODE_NDIM_3:
            return 3
        elif self in _IMAGE_MODE_NDIM_2:
            return 2
        else:
            raise NotImplementedError()

    def to_dtype(self):
        if self in _IMAGE_MODE_DTYPE_UINT8:
            return np.uint8
        elif self in _IMAGE_MODE_DTYPE_FLOAT32:
            return np.float32
        else:
            raise NotImplementedError()

    def to_num_channels(self):
        if self in _IMAGE_MODE_NUM_CHANNELS_4:
            return 4
        elif self in _IMAGE_MODE_NUM_CHANNELS_3:
            return 3
        elif self in _IMAGE_MODE_NUM_CHANNELS_2:
            return None
        else:
            raise NotImplementedError

    def supports_gcn_mode(self):
        return self not in _IMAGE_MODE_NON_GCN_TO_GCN

    def to_gcn_mode(self):
        if not self.supports_gcn_mode():
            raise RuntimeError(f'image_mode={self} not supported.')
        return _IMAGE_MODE_NON_GCN_TO_GCN[self]

    def in_gcn_mode(self):
        return self in _IMAGE_MODE_GCN_TO_NON_GCN

    def to_non_gcn_mode(self):
        if not self.in_gcn_mode():
            raise RuntimeError(f'image_mode={self} not in gcn mode.')
        return _IMAGE_MODE_GCN_TO_NON_GCN[self]


_IMAGE_MODE_NDIM_3 = {
    ImageMode.RGB,
    ImageMode.RGB_GCN,
    ImageMode.RGBA,
    ImageMode.HSV,
    ImageMode.HSV_GCN,
    ImageMode.HSL,
    ImageMode.HSL_GCN,
}

_IMAGE_MODE_NDIM_2 = {
    ImageMode.GRAYSCALE,
    ImageMode.GRAYSCALE_GCN,
}

_IMAGE_MODE_DTYPE_UINT8 = {
    ImageMode.RGB,
    ImageMode.RGBA,
    ImageMode.HSV,
    ImageMode.HSL,
    ImageMode.GRAYSCALE,
}

_IMAGE_MODE_DTYPE_FLOAT32 = {
    ImageMode.RGB_GCN,
    ImageMode.HSV_GCN,
    ImageMode.HSL_GCN,
    ImageMode.GRAYSCALE_GCN,
}

_IMAGE_MODE_NUM_CHANNELS_4 = {
    ImageMode.RGBA,
}

_IMAGE_MODE_NUM_CHANNELS_3 = {
    ImageMode.RGB,
    ImageMode.RGB_GCN,
    ImageMode.HSV,
    ImageMode.HSV_GCN,
    ImageMode.HSL,
    ImageMode.HSL_GCN,
}

_IMAGE_MODE_NUM_CHANNELS_2 = {
    ImageMode.GRAYSCALE,
    ImageMode.GRAYSCALE_GCN,
}

_IMAGE_MODE_NON_GCN_TO_GCN = {
    ImageMode.RGB: ImageMode.RGB_GCN,
    ImageMode.HSV: ImageMode.HSV_GCN,
    ImageMode.HSL: ImageMode.HSL_GCN,
    ImageMode.GRAYSCALE: ImageMode.GRAYSCALE_GCN,
}

_IMAGE_MODE_GCN_TO_NON_GCN = {val: key for key, val in _IMAGE_MODE_NON_GCN_TO_GCN.items()}


@attrs.define
class ImageSetItemConfig:
    value: Union[
        'Image',
        np.ndarray,
        Tuple[int, ...],
        int,
    ]  # yapf: disable
    alpha: Union[np.ndarray, float] = 1.0


class WritableImageContextDecorator(ContextDecorator):

    def __init__(self, image: 'Image'):
        super().__init__()
        self.image = image

    def __enter__(self):
        if self.image.mat.flags.c_contiguous:
            assert not self.image.mat.flags.writeable

        try:
            self.image.mat.flags.writeable = True
        except ValueError:
            # Copy on write.
            object.__setattr__(
                self.image,
                'mat',
                np.array(self.image.mat),
            )
            assert self.image.mat.flags.writeable

    def __exit__(self, *exc):  # type: ignore
        self.image.mat.flags.writeable = False


_IMAGE_SRC_MODE_TO_PRE_SLICE: Mapping[ImageMode, Sequence[int]] = {
    # HSL -> HLS.
    ImageMode.HSL: [0, 2, 1],
}

_IMAGE_SRC_MODE_TO_RGB_CV_CODE = {
    ImageMode.GRAYSCALE: cv.COLOR_GRAY2RGB,
    ImageMode.RGBA: cv.COLOR_RGBA2RGB,
    ImageMode.HSV: cv.COLOR_HSV2RGB_FULL,
    # NOTE: HSL need pre-slicing.
    ImageMode.HSL: cv.COLOR_HLS2RGB_FULL,
}

_IMAGE_INV_DST_MODE_TO_RGB_CV_CODE = {
    ImageMode.GRAYSCALE: cv.COLOR_RGB2GRAY,
    ImageMode.RGBA: cv.COLOR_RGB2RGBA,
    ImageMode.HSV: cv.COLOR_RGB2HSV_FULL,
    # NOTE: HSL need post-slicing.
    ImageMode.HSL: cv.COLOR_RGB2HLS_FULL,
}

_IMAGE_DST_MODE_TO_POST_SLICE: Mapping[ImageMode, Sequence[int]] = {
    # HLS -> HSL.
    ImageMode.HSL: [0, 2, 1],
}

_IMAGE_SRC_DST_MODE_TO_CV_CODE: Mapping[Tuple[ImageMode, ImageMode], int] = {
    (ImageMode.GRAYSCALE, ImageMode.RGBA): cv.COLOR_GRAY2RGBA,
    (ImageMode.RGBA, ImageMode.GRAYSCALE): cv.COLOR_RGBA2GRAY,
}

_E = TypeVar('_E', 'Box', 'Polygon', 'Mask', 'ScoreMap')


@attrs.define(frozen=True, eq=False)
class Image(Shapable):
    mat: np.ndarray
    mode: ImageMode = ImageMode.NONE
    box: Optional['Box'] = None

    def __attrs_post_init__(self):
        if self.mode != ImageMode.NONE:
            # Validate mat.dtype and mode.
            assert self.mode.to_dtype() == self.mat.dtype
            assert self.mode.to_ndim() == self.mat.ndim

        else:
            # Infer image mode based on mat.
            if self.mat.dtype == np.float32:
                raise NotImplementedError('mode is None and mat.dtype == np.float32.')

            elif self.mat.dtype == np.uint8:
                if self.mat.ndim == 2:
                    # Defaults to GRAYSCALE.
                    mode = ImageMode.GRAYSCALE
                elif self.mat.ndim == 3:
                    if self.mat.shape[2] == 4:
                        mode = ImageMode.RGBA
                    elif self.mat.shape[2] == 3:
                        # Defaults to RGB.
                        mode = ImageMode.RGB
                    else:
                        raise NotImplementedError(f'Invalid num_channels={self.mat.shape[2]}.')
                else:
                    raise NotImplementedError(f'mat.ndim={self.mat.ndim} not supported.')

                object.__setattr__(self, 'mode', mode)

            else:
                raise NotImplementedError(f'Invalid mat.dtype={self.mat.dtype}.')

        # For the control of write.
        self.mat.flags.writeable = False

        if self.box and self.shape != self.box.shape:
            raise RuntimeError('self.shape != box.shape.')

    ###############
    # Constructor #
    ###############
    @classmethod
    def from_shape(
        cls,
        shape: Tuple[int, int],
        num_channels: int = 3,
        value: Union[Tuple[int, ...], int] = 255,
    ):
        height, width = shape

        if num_channels == 0:
            mat_shape = (height, width)

        else:
            assert num_channels > 0

            if isinstance(value, tuple):
                assert len(value) == num_channels

            mat_shape = (height, width, num_channels)

        mat = np.full(mat_shape, fill_value=value, dtype=np.uint8)
        return cls(mat=mat)

    @classmethod
    def from_shapable(
        cls,
        shapable: Shapable,
        num_channels: int = 3,
        value: Union[Tuple[int, ...], int] = 255,
    ):
        return cls.from_shape(
            shape=shapable.shape,
            num_channels=num_channels,
            value=value,
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
    def num_channels(self):
        if self.mat.ndim == 2:
            return 0
        else:
            assert self.mat.ndim == 3
            return self.mat.shape[2]

    @property
    def writable_context(self):
        return WritableImageContextDecorator(self)

    ##############
    # Conversion #
    ##############
    @classmethod
    def from_pil_image(cls, pil_image: PilImage.Image):
        # NOTE: Make a copy explicitly, otherwise is not writable.
        mat = np.array(pil_image, dtype=np.uint8)
        return cls(mat=mat)

    def to_pil_image(self):
        return PilImage.fromarray(self.mat)

    @classmethod
    def from_file(cls, path: PathType, disable_exif_orientation: bool = False):
        # NOTE: PilImage.open cannot handle `~`.
        path = io.file(path).expanduser()

        pil_image = PilImage.open(str(path))
        pil_image.load()

        if not disable_exif_orientation:
            # https://exiftool.org/TagNames/EXIF.html
            # https://github.com/python-pillow/Pillow/blob/main/src/PIL/ImageOps.py#L571
            # Avoid unnecessary copy.
            if pil_image.getexif().get(0x0112):
                pil_image = PilImageOps.exif_transpose(pil_image)

        return cls.from_pil_image(pil_image)

    def to_file(self, path: PathType, disable_to_rgb_image: bool = False):
        image = self
        if not disable_to_rgb_image:
            image = image.to_rgb_image()

        pil_image = image.to_pil_image()

        path = io.file(path).expanduser()
        pil_image.save(str(path))

    ############
    # Operator #
    ############
    def copy(self):
        return attrs.evolve(self, mat=self.mat.copy())

    def assign_mat(self, mat: np.ndarray):
        with self.writable_context:
            object.__setattr__(self, 'mat', mat)

    @classmethod
    def check_values_and_alphas_uniqueness(
        cls,
        values: Sequence[Union['Image', np.ndarray, Tuple[int, ...], int, float]],
        alphas: Sequence[Union['ScoreMap', np.ndarray, float]],
    ):
        return check_elements_uniqueness(values) and check_elements_uniqueness(alphas)

    @classmethod
    def unpack_element_value_tuples(
        cls,
        element_value_tuples: Iterable[
            Union[
                Tuple[
                    _E,
                    Union['Image', np.ndarray, Tuple[int, ...], int],
                ],
                Tuple[
                    _E,
                    Union['Image', np.ndarray, Tuple[int, ...], int],
                    Union[float, np.ndarray],
                ],
            ]
        ],
    ):  # yapf: disable
        elements: List[_E] = []
        values: List[Union[Image, np.ndarray, Tuple[int, ...], int]] = []
        alphas: List[Union[float, np.ndarray]] = []

        for element_value_tuple in element_value_tuples:
            if len(element_value_tuple) == 2:
                element, value = element_value_tuple
                alpha = 1.0
            else:
                element, value, alpha = element_value_tuple
            elements.append(element)
            values.append(value)
            alphas.append(alpha)

        return elements, values, alphas

    def fill_by_box_value_tuples(
        self,
        box_value_tuples: Iterable[
            Union[
                Tuple[
                    'Box',
                    Union['Image', np.ndarray, Tuple[int, ...], int],
                ],
                Tuple[
                    'Box',
                    Union['Image', np.ndarray, Tuple[int, ...], int],
                    Union[float, np.ndarray],
                ],
            ]
        ],
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        skip_values_uniqueness_check: bool = False,
    ):  # yapf: disable
        boxes, values, alphas = self.unpack_element_value_tuples(box_value_tuples)

        boxes_mask = generate_fill_by_boxes_mask(self.shape, boxes, mode)
        if boxes_mask is None:
            for box, value, alpha in zip(boxes, values, alphas):
                box.fill_image(
                    image=self,
                    value=value,
                    alpha=alpha,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = self.check_values_and_alphas_uniqueness(values, alphas)

            if unique:
                boxes_mask.fill_image(
                    image=self,
                    value=values[0],
                    alpha=alphas[0],
                )
            else:
                for box, value, alpha in zip(boxes, values, alphas):
                    box_mask = box.extract_mask(boxes_mask).to_box_attached(box)
                    box_mask.fill_image(
                        image=self,
                        value=value,
                        alpha=alpha,
                    )

    def fill_by_boxes(
        self,
        boxes: Iterable['Box'],
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        alpha: Union[np.ndarray, float] = 1.0,
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
    ):
        self.fill_by_box_value_tuples(
            box_value_tuples=((box, value, alpha) for box in boxes),
            mode=mode,
            skip_values_uniqueness_check=True,
        )

    def fill_by_polygon_value_tuples(
        self,
        polygon_value_tuples: Iterable[
            Union[
                Tuple[
                    'Polygon',
                    Union['Image', np.ndarray, Tuple[int, ...], int],
                ],
                Tuple[
                    'Polygon',
                    Union['Image', np.ndarray, Tuple[int, ...], int],
                    Union[float, np.ndarray],
                ],
            ]
        ],
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        skip_values_uniqueness_check: bool = False,
    ):  # yapf: disable
        polygons, values, alphas = self.unpack_element_value_tuples(polygon_value_tuples)

        polygons_mask = generate_fill_by_polygons_mask(self.shape, polygons, mode)
        if polygons_mask is None:
            for polygon, value, alpha in zip(polygons, values, alphas):
                polygon.fill_image(
                    image=self,
                    value=value,
                    alpha=alpha,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = self.check_values_and_alphas_uniqueness(values, alphas)

            if unique:
                polygons_mask.fill_image(
                    image=self,
                    value=values[0],
                    alpha=alphas[0],
                )
            else:
                for polygon, value, alpha in zip(polygons, values, alphas):
                    bounding_box = polygon.to_bounding_box()
                    polygon_mask = bounding_box.extract_mask(polygons_mask)
                    polygon_mask = polygon_mask.to_box_attached(bounding_box)
                    polygon_mask.fill_image(
                        image=self,
                        value=value,
                        alpha=alpha,
                    )

    def fill_by_polygons(
        self,
        polygons: Iterable['Polygon'],
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        alpha: Union[np.ndarray, float] = 1.0,
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
    ):
        self.fill_by_polygon_value_tuples(
            polygon_value_tuples=((polygon, value, alpha) for polygon in polygons),
            mode=mode,
            skip_values_uniqueness_check=True,
        )

    def fill_by_mask_value_tuples(
        self,
        mask_value_tuples: Iterable[
            Union[
                Tuple[
                    'Mask',
                    Union['Image', np.ndarray, Tuple[int, ...], int],
                ],
                Tuple[
                    'Mask',
                    Union['Image', np.ndarray, Tuple[int, ...], int],
                    Union[float, np.ndarray],
                ],
            ]
        ],
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        skip_values_uniqueness_check: bool = False,
    ):  # yapf: disable
        masks, values, alphas = self.unpack_element_value_tuples(mask_value_tuples)

        masks_mask = generate_fill_by_masks_mask(self.shape, masks, mode)
        if masks_mask is None:
            for mask, value, alpha in zip(masks, values, alphas):
                mask.fill_image(
                    image=self,
                    value=value,
                    alpha=alpha,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = self.check_values_and_alphas_uniqueness(values, alphas)

            if unique:
                masks_mask.fill_image(
                    image=self,
                    value=values[0],
                    alpha=alphas[0],
                )
            else:
                for mask, value, alpha in zip(masks, values, alphas):
                    if mask.box:
                        boxed_mask = mask.box.extract_mask(masks_mask)
                    else:
                        boxed_mask = masks_mask

                    boxed_mask = boxed_mask.copy()
                    mask.to_inverted_mask().fill_mask(boxed_mask, value=0)
                    boxed_mask.fill_image(
                        image=self,
                        value=value,
                        alpha=alpha,
                    )

    def fill_by_masks(
        self,
        masks: Iterable['Mask'],
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        alpha: Union[np.ndarray, float] = 1.0,
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
    ):
        self.fill_by_mask_value_tuples(
            mask_value_tuples=((mask, value, alpha) for mask in masks),
            mode=mode,
            skip_values_uniqueness_check=True,
        )

    def fill_by_score_map_value_tuples(
        self,
        score_map_value_tuples: Iterable[
            Tuple[
                'ScoreMap',
                Union['Image', np.ndarray, Tuple[int, ...], int],
            ]
        ],
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
        skip_values_uniqueness_check: bool = False,
    ):  # yapf: disable
        # NOTE: score maps serve as masks & alphas, hence ignoring unpacked alphas.
        score_maps, values, _ = self.unpack_element_value_tuples(score_map_value_tuples)

        score_maps_mask = generate_fill_by_score_maps_mask(self.shape, score_maps, mode)
        if score_maps_mask is None:
            for score_map, value in zip(score_maps, values):
                score_map.fill_image(
                    image=self,
                    value=value,
                )

        else:
            unique = True
            if not skip_values_uniqueness_check:
                unique = check_elements_uniqueness(values)

            if unique:
                # This is unlikely to happen.
                score_maps_mask.fill_image(
                    image=self,
                    value=values[0],
                    alpha=score_maps[0],
                )
            else:
                for score_map, value in zip(score_maps, values):
                    if score_map.box:
                        boxed_mask = score_map.box.extract_mask(score_maps_mask)
                    else:
                        boxed_mask = score_maps_mask

                    boxed_mask = boxed_mask.copy()
                    score_map.to_mask().to_inverted_mask().fill_mask(boxed_mask, value=0)
                    boxed_mask.fill_image(
                        image=self,
                        value=value,
                        alpha=score_map,
                    )

    def fill_by_score_maps(
        self,
        score_maps: Iterable['ScoreMap'],
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        mode: ElementSetOperationMode = ElementSetOperationMode.UNION,
    ):
        self.fill_by_score_map_value_tuples(
            score_map_value_tuples=((score_map, value) for score_map in score_maps),
            mode=mode,
            skip_values_uniqueness_check=True,
        )

    def __setitem__(
        self,
        element: Union[
            'Box',
            'Polygon',
            'Mask',
            'ScoreMap',
        ],
        config: Union[
            'Image',
            np.ndarray,
            Tuple[int, ...],
            int,
            ImageSetItemConfig,
        ],
    ):  # yapf: disable
        if not isinstance(config, ImageSetItemConfig):
            value = config
            alpha = 1.0
        else:
            assert isinstance(config, ImageSetItemConfig)
            value = config.value
            alpha = config.alpha

        if isinstance(value, tuple):
            assert value
            assert isinstance(value[0], int)
        else:
            assert not isinstance(value, abc.Iterable)

        # Type inference cannot handle this case.
        value = cast(
            Union[
                'Image',
                np.ndarray,
                Tuple[int, ...],
                int,
            ],
            value
        )  # yapf: disable
        assert not isinstance(alpha, abc.Iterable)

        if isinstance(element, ScoreMap):
            element.fill_image(image=self, value=value)
        else:
            element.fill_image(image=self, value=value, alpha=alpha)

    def __getitem__(
        self,
        element: Union[
            'Box',
            'Polygon',
            'Mask',
        ],
    ):  # yapf: disable
        return element.extract_image(self)

    def to_box_attached(self, box: 'Box'):
        assert self.height == box.height
        assert self.width == box.width
        return attrs.evolve(self, box=box)

    def to_box_detached(self):
        assert self.box
        return attrs.evolve(self, box=None)

    def to_gcn_image(
        self,
        lamb: float = 0,
        eps: float = 1E-8,
        scale: float = 1.0,
    ):
        # Global contrast normalization.
        # https://cedar.buffalo.edu/~srihari/CSE676/12.2%20Computer%20Vision.pdf
        # (H, W) or (H, W, 3)
        mode = self.mode.to_gcn_mode()

        mat = self.mat.astype(np.float32)

        # Normalize mean(contrast).
        mean = np.mean(mat)
        mat -= mean

        # Std normalized.
        std = np.sqrt(lamb + np.mean(mat**2))
        mat /= max(eps, std)
        if scale != 1.0:
            mat *= scale

        return Image(mat=mat, mode=mode)

    def to_non_gcn_image(self):
        mode = self.mode.to_non_gcn_mode()

        assert self.mat.dtype == np.float32
        val_min = np.min(self.mat)
        mat = self.mat - val_min
        gap = np.max(mat)
        mat = mat / gap * 255.0
        mat = np.round(mat)
        mat = np.clip(mat, 0, 255).astype(np.uint8)

        return Image(mat=mat, mode=mode)  # type: ignore

    def to_target_mode_image(self, target_mode: ImageMode):
        if target_mode == self.mode:
            # Identity.
            return self

        skip_copy = False
        if self.mode.in_gcn_mode():
            self = self.to_non_gcn_image()
            skip_copy = True

        if self.mode == target_mode:
            # GCN to non-GCN conversion.
            return self if skip_copy else self.copy()

        mat = self.mat

        # Pre-slicing.
        if self.mode in _IMAGE_SRC_MODE_TO_PRE_SLICE:
            mat: np.ndarray = mat[:, :, _IMAGE_SRC_MODE_TO_PRE_SLICE[self.mode]]

        if (self.mode, target_mode) in _IMAGE_SRC_DST_MODE_TO_CV_CODE:
            # Shortcut.
            cv_code = _IMAGE_SRC_DST_MODE_TO_CV_CODE[(self.mode, target_mode)]
            dst_mat: np.ndarray = cv.cvtColor(mat, cv_code)
            return Image(mat=dst_mat, mode=target_mode)

        dst_mat = mat
        if self.mode != ImageMode.RGB:
            # Convert to RGB.
            dst_mat: np.ndarray = cv.cvtColor(mat, _IMAGE_SRC_MODE_TO_RGB_CV_CODE[self.mode])

        if target_mode == ImageMode.RGB:
            # No need to continue.
            return Image(mat=dst_mat, mode=ImageMode.RGB)

        # Convert RGB to target mode.
        assert target_mode in _IMAGE_INV_DST_MODE_TO_RGB_CV_CODE
        dst_mat: np.ndarray = cv.cvtColor(dst_mat, _IMAGE_INV_DST_MODE_TO_RGB_CV_CODE[target_mode])

        # Post-slicing.
        if target_mode in _IMAGE_DST_MODE_TO_POST_SLICE:
            dst_mat: np.ndarray = dst_mat[:, :, _IMAGE_DST_MODE_TO_POST_SLICE[target_mode]]

        return Image(mat=dst_mat, mode=target_mode)

    def to_grayscale_image(self):
        return self.to_target_mode_image(ImageMode.GRAYSCALE)

    def to_rgb_image(self):
        return self.to_target_mode_image(ImageMode.RGB)

    def to_rgba_image(self):
        return self.to_target_mode_image(ImageMode.RGBA)

    def to_hsv_image(self):
        return self.to_target_mode_image(ImageMode.HSV)

    def to_hsl_image(self):
        return self.to_target_mode_image(ImageMode.HSL)

    def to_shifted_image(self, offset_y: int = 0, offset_x: int = 0):
        assert self.box
        shifted_box = self.box.to_shifted_box(offset_y=offset_y, offset_x=offset_x)
        return attrs.evolve(self, box=shifted_box)

    def to_resized_image(
        self,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
        cv_resize_interpolation: int = cv.INTER_CUBIC,
    ):
        _, _, resized_height, resized_width = generate_shape_and_resized_shape(
            shapable_or_shape=self,
            resized_height=resized_height,
            resized_width=resized_width,
        )
        mat = cv.resize(
            self.mat,
            (resized_width, resized_height),
            interpolation=cv_resize_interpolation,
        )
        return attrs.evolve(self, mat=mat)

    def to_conducted_resized_image(
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
        resized_image = self.to_box_detached().to_resized_image(
            resized_height=resized_box.height,
            resized_width=resized_box.width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_image = resized_image.to_box_attached(resized_box)
        return resized_image

    def to_cropped_image(
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


# Cyclic dependency, by design.
from .uniqueness import check_elements_uniqueness  # noqa: E402
from .box import Box, generate_fill_by_boxes_mask  # noqa: E402
from .polygon import Polygon, generate_fill_by_polygons_mask  # noqa: E402
from .mask import Mask, generate_fill_by_masks_mask  # noqa: E402
from .score_map import ScoreMap, generate_fill_by_score_maps_mask  # noqa: E402
