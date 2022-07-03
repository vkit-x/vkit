from typing import Optional, Tuple, Mapping, Any

import attrs
import numpy as np
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.element import Image
from ..interface import DistortionConfig, DistortionNopState, Distortion
from .opt import to_rgb_image, to_original_image, clip_mat_back_to_uint8


def _estimate_gaussian_kernel_size(sigma: float):
    kernel_size = max(3, round(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _get_anti_aliasing_kernel_size_and_padding(anti_aliasing_sigma: float):
    kernel_size = _estimate_gaussian_kernel_size(anti_aliasing_sigma)

    anti_aliasing_ksize = (kernel_size, kernel_size)
    anti_aliasing_kernel_padding = kernel_size // 2 * 2
    return anti_aliasing_ksize, anti_aliasing_kernel_padding


def _apply_anti_aliasing_to_kernel(
    kernel: np.ndarray,
    anti_aliasing_ksize: Tuple[int, int],
    anti_aliasing_sigma: float,
):
    return cv.GaussianBlur(kernel, anti_aliasing_ksize, anti_aliasing_sigma)


@attrs.define
class GaussianBlurConfig(DistortionConfig):
    sigma: float


def gaussian_blur_image(
    config: GaussianBlurConfig,
    state: Optional[DistortionNopState[GaussianBlurConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    kind = image.kind
    image = to_rgb_image(image, kind)

    kernel_size = _estimate_gaussian_kernel_size(config.sigma)
    ksize = (kernel_size, kernel_size)
    mat = cv.GaussianBlur(image.mat, ksize, config.sigma)
    image = attrs.evolve(image, mat=mat)

    image = to_original_image(image, kind)
    return image


gaussian_blur = Distortion(
    config_cls=GaussianBlurConfig,
    state_cls=DistortionNopState[GaussianBlurConfig],
    func_image=gaussian_blur_image,
)


@attrs.define
class DefocusBlurConfig(DistortionConfig):
    radius: int
    anti_aliasing_sigma: float = 0.5


def defocus_blur_image(
    config: DefocusBlurConfig,
    state: Optional[DistortionNopState[DefocusBlurConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    # Generate blurring kernel.
    assert 0 < config.radius
    kernel_size = 2 * config.radius + 1

    (
        anti_aliasing_ksize,
        anti_aliasing_kernel_padding,
    ) = _get_anti_aliasing_kernel_size_and_padding(config.anti_aliasing_sigma)
    kernel_size += anti_aliasing_kernel_padding

    begin = -(kernel_size // 2)
    end = begin + kernel_size
    coords = np.arange(begin, end)
    x, y = np.meshgrid(coords, coords)
    kernel: np.ndarray = ((x**2 + y**2) <= config.radius**2).astype(np.float32)
    kernel /= kernel.sum()

    kernel = _apply_anti_aliasing_to_kernel(
        kernel,
        anti_aliasing_ksize,
        config.anti_aliasing_sigma,
    )

    # Convolution.
    kind = image.kind
    image = to_rgb_image(image, kind)

    mat = cv.filter2D(image.mat, -1, kernel)
    image = attrs.evolve(image, mat=mat)

    image = to_original_image(image, kind)
    return image


defocus_blur = Distortion(
    config_cls=DefocusBlurConfig,
    state_cls=DistortionNopState[DefocusBlurConfig],
    func_image=defocus_blur_image,
)


@attrs.define
class MotionBlurConfig(DistortionConfig):
    radius: int
    angle: int
    anti_aliasing_sigma: float = 0.5


def motion_blur_image(
    config: MotionBlurConfig,
    state: Optional[DistortionNopState[MotionBlurConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    # Generate blurring kernel.
    kernel_size = 2 * config.radius + 1

    (
        anti_aliasing_ksize,
        anti_aliasing_kernel_padding,
    ) = _get_anti_aliasing_kernel_size_and_padding(config.anti_aliasing_sigma)
    anti_aliasing_kernel_padding_half = anti_aliasing_kernel_padding // 2
    center = config.radius + anti_aliasing_kernel_padding_half
    left = anti_aliasing_kernel_padding_half
    right = left + kernel_size - 1
    kernel_size += anti_aliasing_kernel_padding

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[center, left:right + 1] = 1.0

    trans_mat = cv.getRotationMatrix2D(
        (center, center),
        # 1. to [0, 359].
        # 2. getRotationMatrix2D accepts counter-clockwise angle, hence need to subtract.
        360 - (config.angle % 360),
        1.0,
    )
    kernel = cv.warpAffine(kernel, trans_mat, kernel.shape)
    kernel /= kernel.sum()

    kernel = _apply_anti_aliasing_to_kernel(
        kernel,
        anti_aliasing_ksize,
        config.anti_aliasing_sigma,
    )

    # Convolution.
    kind = image.kind
    image = to_rgb_image(image, kind)

    mat = cv.filter2D(image.mat, -1, kernel)
    image = attrs.evolve(image, mat=mat)

    image = to_original_image(image, kind)
    return image


motion_blur = Distortion(
    config_cls=MotionBlurConfig,
    state_cls=DistortionNopState[MotionBlurConfig],
    func_image=motion_blur_image,
)


@attrs.define
class GlassBlurConfig(DistortionConfig):
    sigma: float
    delta: int = 1
    loop: int = 5

    _rng_state: Optional[Mapping[str, Any]] = None

    @property
    def supports_rng_state(self) -> bool:
        return True

    @property
    def rng_state(self) -> Optional[Mapping[str, Any]]:
        return self._rng_state

    @rng_state.setter
    def rng_state(self, val: Mapping[str, Any]):
        self._rng_state = val


def glass_blur_image(
    config: GlassBlurConfig,
    state: Optional[DistortionNopState[GlassBlurConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    kind = image.kind
    image = to_rgb_image(image, kind)

    # Gaussian blur.
    kernel_size = _estimate_gaussian_kernel_size(config.sigma)
    ksize = (kernel_size, kernel_size)
    mat = cv.GaussianBlur(image.mat, ksize, config.sigma)

    # Random pixel swap.
    assert rng is not None

    coords_y = np.arange(image.height)
    coords_x = np.arange(image.width)
    pos_x, pos_y = np.meshgrid(coords_x, coords_y)

    for _ in range(config.loop):
        offset_y = rng.integers(0, 2 * config.delta + 1)
        center_y = np.arange(offset_y, image.height - config.delta, 2 * config.delta + 1)
        num_center_y = center_y.shape[0]
        center_y = center_y.reshape(-1, 1)

        offset_x = rng.integers(0, 2 * config.delta + 1)
        center_x = np.arange(offset_x, image.width - config.delta, 2 * config.delta + 1)
        num_center_x = center_x.shape[0]
        center_x = center_x.reshape(1, -1)

        delta_shape = (num_center_y, num_center_x)
        delta_y = rng.integers(-config.delta, config.delta + 1, delta_shape)
        delta_x = rng.integers(-config.delta, config.delta + 1, delta_shape)

        deformed_y: np.ndarray = pos_y[center_y, center_x] + delta_y
        deformed_y = np.clip(deformed_y, 0, image.height - 1)
        deformed_x: np.ndarray = pos_x[center_y, center_x] + delta_x
        deformed_x = np.clip(deformed_x, 0, image.width - 1)

        # Swap.
        pos_y[center_y, center_x], pos_y[deformed_y, deformed_x] = \
            pos_y[deformed_y, deformed_x], pos_y[center_y, center_x]

        pos_x[center_y, center_x], pos_x[deformed_y, deformed_x] = \
            pos_x[deformed_y, deformed_x], pos_x[center_y, center_x]

    mat = mat[pos_y, pos_x]
    image = attrs.evolve(image, mat=mat)

    image = to_original_image(image, kind)
    return image


glass_blur = Distortion(
    config_cls=GlassBlurConfig,
    state_cls=DistortionNopState[GlassBlurConfig],
    func_image=glass_blur_image,
)


@attrs.define
class ZoomInBlurConfig(DistortionConfig):
    ratio: float = 0.1
    step: float = 0.01
    alpha: float = 0.5


def zoom_in_blur_image(
    config: ZoomInBlurConfig,
    state: Optional[DistortionNopState[ZoomInBlurConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    kind = image.kind
    image = to_rgb_image(image, kind)

    mat = image.mat.astype(np.uint16)
    count = 1
    for ratio in np.arange(
        1 + config.step,
        1 + config.ratio + config.step,
        config.step,
    ):
        resized_height = round(image.height * ratio)
        resized_width = round(image.width * ratio)
        resized_image = image.to_resized_image(resized_height, resized_width)

        pad_y = resized_height - image.height
        up = pad_y // 2
        down = up + image.height - 1

        pad_x = resized_width - image.width
        left = pad_x // 2
        right = left + image.width - 1

        mat += resized_image.mat[up:down + 1, left:right + 1]
        count += 1

    # NOTE: Label confusion could be significant if alpha is large.
    mat: np.ndarray = (1 - config.alpha) * image.mat + config.alpha * np.round(mat / count)
    mat = clip_mat_back_to_uint8(mat)

    image = attrs.evolve(image, mat=mat)

    image = to_original_image(image, kind)
    return image


zoom_in_blur = Distortion(
    config_cls=ZoomInBlurConfig,
    state_cls=DistortionNopState[ZoomInBlurConfig],
    func_image=zoom_in_blur_image,
)
