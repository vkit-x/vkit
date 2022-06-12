from typing import Dict, Sequence, Tuple, Union, Optional
from enum import Enum, unique

import attrs
import numpy as np
from PIL import (
    Image as PilImage,
    ImageOps as PilImageOps,
)
import cv2 as cv

from vkit.utility import PathType
from .type import Shapable
from .opt import generate_shape_and_resized_shape


@unique
class ImageKind(Enum):
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

    @staticmethod
    def to_ndim(image_kind: 'ImageKind'):
        return _image_kind_to_ndim(image_kind)

    @staticmethod
    def to_dtype(image_kind: 'ImageKind'):
        return _image_kind_to_dtype(image_kind)

    @staticmethod
    def to_num_channels(image_kind: 'ImageKind'):
        return _image_kind_to_num_channels(image_kind)

    @staticmethod
    def supports_gcn_mode(image_kind: 'ImageKind'):
        _image_kind_supports_gcn_mode(image_kind)

    @staticmethod
    def to_gcn_mode(image_kind: 'ImageKind'):
        return _image_kind_to_gcn_mode(image_kind)

    @staticmethod
    def in_gcn_mode(image_kind: 'ImageKind'):
        return _image_kind_in_gcn_mode(image_kind)

    @staticmethod
    def to_non_gcn_mode(image_kind: 'ImageKind'):
        return _image_kind_to_non_gcn_mode(image_kind)


_IMAGE_KIND_NDIM_3 = {
    ImageKind.RGB,
    ImageKind.RGB_GCN,
    ImageKind.RGBA,
    ImageKind.HSV,
    ImageKind.HSV_GCN,
    ImageKind.HSL,
    ImageKind.HSL_GCN,
}

_IMAGE_KIND_NDIM_2 = {
    ImageKind.GRAYSCALE,
    ImageKind.GRAYSCALE_GCN,
}


def _image_kind_to_ndim(image_kind: ImageKind):
    if image_kind in _IMAGE_KIND_NDIM_3:
        return 3

    elif image_kind in _IMAGE_KIND_NDIM_2:
        return 2

    else:
        raise NotImplementedError()


_IMAGE_KIND_DTYPE_UINT8 = {
    ImageKind.RGB,
    ImageKind.RGBA,
    ImageKind.HSV,
    ImageKind.HSL,
    ImageKind.GRAYSCALE,
}

_IMAGE_KIND_DTYPE_FLOAT32 = {
    ImageKind.RGB_GCN,
    ImageKind.HSV_GCN,
    ImageKind.HSL_GCN,
    ImageKind.GRAYSCALE_GCN,
}


def _image_kind_to_dtype(image_kind: ImageKind):
    if image_kind in _IMAGE_KIND_DTYPE_UINT8:
        return np.uint8

    elif image_kind in _IMAGE_KIND_DTYPE_FLOAT32:
        return np.float32

    else:
        raise NotImplementedError()


_IMAGE_KIND_NUM_CHANNELS_4 = {
    ImageKind.RGBA,
}

_IMAGE_KIND_NUM_CHANNELS_3 = {
    ImageKind.RGB,
    ImageKind.RGB_GCN,
    ImageKind.HSV,
    ImageKind.HSV_GCN,
    ImageKind.HSL,
    ImageKind.HSL_GCN,
}

_IMAGE_KIND_NUM_CHANNELS_2 = {
    ImageKind.GRAYSCALE,
    ImageKind.GRAYSCALE_GCN,
}


def _image_kind_to_num_channels(image_kind: ImageKind):
    if image_kind in _IMAGE_KIND_NUM_CHANNELS_4:
        return 4

    elif image_kind in _IMAGE_KIND_NUM_CHANNELS_3:
        return 3

    elif image_kind in _IMAGE_KIND_NUM_CHANNELS_2:
        return None

    else:
        raise NotImplementedError


_IMAGE_KIND_NON_GCN_TO_GCN = {
    ImageKind.RGB: ImageKind.RGB_GCN,
    ImageKind.HSV: ImageKind.HSV_GCN,
    ImageKind.HSL: ImageKind.HSL_GCN,
    ImageKind.GRAYSCALE: ImageKind.GRAYSCALE_GCN,
}


def _image_kind_supports_gcn_mode(image_kind: ImageKind):
    return image_kind not in _IMAGE_KIND_NON_GCN_TO_GCN


def _image_kind_to_gcn_mode(image_kind: ImageKind):
    if _image_kind_supports_gcn_mode(image_kind):
        raise RuntimeError(f'image_kind={image_kind} not supported.')
    return _IMAGE_KIND_NON_GCN_TO_GCN[image_kind]


_IMAGE_KIND_GCN_TO_NON_GCN = {val: key for key, val in _IMAGE_KIND_NON_GCN_TO_GCN.items()}


def _image_kind_in_gcn_mode(image_kind: ImageKind):
    return image_kind in _IMAGE_KIND_GCN_TO_NON_GCN


def _image_kind_to_non_gcn_mode(image_kind: ImageKind):
    if not _image_kind_in_gcn_mode(image_kind):
        raise RuntimeError(f'image_kind={image_kind} not supported.')
    return _IMAGE_KIND_GCN_TO_NON_GCN[image_kind]


_IMAGE_SRC_KIND_TO_PRE_SLICE: Dict[ImageKind, Sequence[int]] = {
    # HSL -> HLS.
    ImageKind.HSL: [0, 2, 1],
}

_IMAGE_SRC_KIND_TO_RGB_CV_CODE = {
    ImageKind.GRAYSCALE: cv.COLOR_GRAY2RGB,
    ImageKind.RGBA: cv.COLOR_RGBA2RGB,
    ImageKind.HSV: cv.COLOR_HSV2RGB_FULL,
    # NOTE: HSL need pre-slicing.
    ImageKind.HSL: cv.COLOR_HLS2RGB_FULL,
}

_IMAGE_INV_DST_KIND_TO_RGB_CV_CODE = {
    ImageKind.GRAYSCALE: cv.COLOR_RGB2GRAY,
    ImageKind.RGBA: cv.COLOR_RGB2RGBA,
    ImageKind.HSV: cv.COLOR_RGB2HSV_FULL,
    # NOTE: HSL need post-slicing.
    ImageKind.HSL: cv.COLOR_RGB2HLS_FULL,
}

_IMAGE_DST_KIND_TO_POST_SLICE: Dict[ImageKind, Sequence[int]] = {
    # HLS -> HSL.
    ImageKind.HSL: [0, 2, 1],
}

_IMAGE_SRC_DST_KIND_TO_CV_CODE: Dict[Tuple[ImageKind, ImageKind], int] = {
    (ImageKind.GRAYSCALE, ImageKind.RGBA): cv.COLOR_GRAY2RGBA,
    (ImageKind.RGBA, ImageKind.GRAYSCALE): cv.COLOR_RGBA2GRAY,
}


@attrs.define
class Image(Shapable):
    mat: np.ndarray
    kind: ImageKind = ImageKind.NONE
    box: Optional['Box'] = None

    def __attrs_post_init__(self):
        if self.kind != ImageKind.NONE:
            # Validate mat.dtype and kind.
            assert ImageKind.to_dtype(self.kind) == self.mat.dtype
            assert ImageKind.to_ndim(self.kind) == self.mat.ndim

        else:
            # Infer image kind based on mat.
            if self.mat.dtype == np.float32:
                raise NotImplementedError('kind is None and mat.dtype == np.float32.')

            elif self.mat.dtype == np.uint8:
                if self.mat.ndim == 2:
                    # Defaults to GRAYSCALE.
                    self.kind = ImageKind.GRAYSCALE
                elif self.mat.ndim == 3:
                    if self.mat.shape[2] == 4:
                        self.kind = ImageKind.RGBA
                    elif self.mat.shape[2] == 3:
                        # Defaults to RGB.
                        self.kind = ImageKind.RGB
                    else:
                        raise NotImplementedError(f'Invalid num_channels={self.mat.shape[2]}.')

            else:
                raise NotImplementedError(f'Invalid mat.dtype={self.mat.dtype}.')

        if self.box and self.shape != self.box.shape:
            raise RuntimeError('self.shape != box.shape.')

    ###############
    # Constructor #
    ###############
    @staticmethod
    def from_shape(
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
        return Image(mat=mat)

    @staticmethod
    def from_shapable(
        shapable: Shapable,
        num_channels: int = 3,
        value: Union[Tuple[int, ...], int] = 255,
    ):
        return Image.from_shape(
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

    ##############
    # Conversion #
    ##############
    @staticmethod
    def from_pil_image(pil_image: PilImage.Image):
        # NOTE: Make a copy explicitly.
        mat = np.array(pil_image, dtype=np.uint8)
        return Image(mat=mat)

    def to_pil_image(self):
        return PilImage.fromarray(self.mat)

    @staticmethod
    def from_file(path: PathType, disable_exif_orientation: bool = False):
        pil_img = PilImage.open(str(path))
        pil_img.load()

        if not disable_exif_orientation:
            # https://exiftool.org/TagNames/EXIF.html
            # https://github.com/python-pillow/Pillow/blob/main/src/PIL/ImageOps.py#L571
            # Avoid unnecessary copy.
            if pil_img.getexif().get(0x0112):
                pil_img = PilImageOps.exif_transpose(pil_img)

        return Image.from_pil_image(pil_img)

    def to_file(self, path: PathType, disable_to_rgb_image: bool = False):
        image = self
        if not disable_to_rgb_image:
            image = image.to_rgb_image()

        pil_img = image.to_pil_image()
        pil_img.save(str(path))

    ############
    # Operator #
    ############
    def copy(self):
        mat = self.mat.copy()
        return attrs.evolve(self, mat=mat)

    def to_gcn_image(
        self,
        lamb: float = 0,
        eps: float = 1E-8,
        scale: float = 1.0,
    ):
        # Global contrast normalization.
        # https://cedar.buffalo.edu/~srihari/CSE676/12.2%20Computer%20Vision.pdf
        # (H, W) or (H, W, 3)
        kind = ImageKind.to_gcn_mode(self.kind)

        mat = self.mat.astype(np.float32)

        # Normalize mean(contrast).
        mean = np.mean(mat)
        mat -= mean

        # Std normalized.
        std = np.sqrt(lamb + np.mean(mat**2))
        mat /= max(eps, std)
        if scale != 1.0:
            mat *= scale

        return Image(mat=mat, kind=kind)

    def to_non_gcn_image(self):
        kind = ImageKind.to_non_gcn_mode(self.kind)

        assert self.mat.dtype == np.float32
        val_min = np.min(self.mat)
        mat = self.mat - val_min
        gap = np.max(mat)
        mat = mat / gap * 255.0
        mat = np.round(mat)
        mat = np.clip(mat, 0, 255).astype(np.uint8)

        return Image(mat=mat, kind=kind)

    def to_target_kind_image(self, target_kind: ImageKind):
        if target_kind == self.kind:
            # Identity.
            return self

        skip_copy = False
        if ImageKind.in_gcn_mode(self.kind):
            self = self.to_non_gcn_image()
            skip_copy = True

        if self.kind == target_kind:
            # GCN to non-GCN conversion.
            return self if skip_copy else self.copy()

        mat = self.mat

        # Pre-slicing.
        if self.kind in _IMAGE_SRC_KIND_TO_PRE_SLICE:
            mat: np.ndarray = mat[:, :, _IMAGE_SRC_KIND_TO_PRE_SLICE[self.kind]]

        if (self.kind, target_kind) in _IMAGE_SRC_DST_KIND_TO_CV_CODE:
            # Shortcut.
            cv_code = _IMAGE_SRC_DST_KIND_TO_CV_CODE[(self.kind, target_kind)]
            dst_mat: np.ndarray = cv.cvtColor(mat, cv_code)
            return Image(mat=dst_mat, kind=target_kind)

        dst_mat = mat
        if self.kind != ImageKind.RGB:
            # Convert to RGB.
            dst_mat: np.ndarray = cv.cvtColor(mat, _IMAGE_SRC_KIND_TO_RGB_CV_CODE[self.kind])

        if target_kind == ImageKind.RGB:
            # No need to continue.
            return Image(mat=dst_mat, kind=ImageKind.RGB)

        # Convert RGB to target kind.
        assert target_kind in _IMAGE_INV_DST_KIND_TO_RGB_CV_CODE
        dst_mat: np.ndarray = cv.cvtColor(dst_mat, _IMAGE_INV_DST_KIND_TO_RGB_CV_CODE[target_kind])

        # Post-slicing.
        if target_kind in _IMAGE_DST_KIND_TO_POST_SLICE:
            dst_mat: np.ndarray = dst_mat[:, :, _IMAGE_DST_KIND_TO_POST_SLICE[target_kind]]

        return Image(mat=dst_mat, kind=target_kind)

    def to_grayscale_image(self):
        return self.to_target_kind_image(ImageKind.GRAYSCALE)

    def to_rgb_image(self):
        return self.to_target_kind_image(ImageKind.RGB)

    def to_rgba_image(self):
        return self.to_target_kind_image(ImageKind.RGBA)

    def to_hsv_image(self):
        return self.to_target_kind_image(ImageKind.HSV)

    def to_hsl_image(self):
        return self.to_target_kind_image(ImageKind.HSL)

    def to_shifted_image(self, y_offset: int = 0, x_offset: int = 0):
        assert self.box
        shifted_box = self.box.to_shifted_box(y_offset=y_offset, x_offset=x_offset)
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
        resized_image = self.to_resized_image(
            resized_height=resized_box.height,
            resized_width=resized_box.width,
            cv_resize_interpolation=cv_resize_interpolation,
        )
        resized_image.box = resized_box
        return resized_image


# Cyclic dependency, by design.
from .box import Box  # noqa: E402
