from typing import Optional, Tuple, Mapping, Any

import attrs
import numpy as np
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.element import Image, ImageKind
from ..interface import DistortionConfig, DistortionNopState, Distortion
from .opt import to_rgb_image, to_original_image, clip_mat_back_to_uint8


@attrs.define
class JpegQualityConfig(DistortionConfig):
    quality: int


def jpeg_quality_image(
    config: JpegQualityConfig,
    state: Optional[DistortionNopState[JpegQualityConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    kind = image.kind
    image = to_rgb_image(image, kind)

    assert 0 <= config.quality <= 100
    _, buffer = cv.imencode('.jpeg', image.mat, [cv.IMWRITE_JPEG_QUALITY, config.quality])
    mat: np.ndarray = cv.imdecode(buffer, cv.IMREAD_UNCHANGED)
    image = attrs.evolve(image, mat=mat)

    image = to_original_image(image, kind)
    return image


jpeg_quality = Distortion(
    config_cls=JpegQualityConfig,
    state_cls=DistortionNopState[JpegQualityConfig],
    func_image=jpeg_quality_image,
)


@attrs.define
class PixelationConfig(DistortionConfig):
    ratio: float


def pixelation_image(
    config: PixelationConfig,
    state: Optional[DistortionNopState[PixelationConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    # Downsample.
    assert 0 < config.ratio < 1
    resized_height = round(image.height * config.ratio)
    resized_width = round(image.width * config.ratio)
    dsize = (resized_width, resized_height)
    mat: np.ndarray = cv.resize(image.mat, dsize, interpolation=cv.INTER_LINEAR)

    # Upsample.
    dsize = (image.width, image.height)
    mat: np.ndarray = cv.resize(mat, dsize, interpolation=cv.INTER_NEAREST)

    image = attrs.evolve(image, mat=mat)
    return image


pixelation = Distortion(
    config_cls=PixelationConfig,
    state_cls=DistortionNopState[PixelationConfig],
    func_image=pixelation_image,
)


def generate_diamond_square_mask(
    shape: Tuple[int, int],
    roughness: float,
    rng: RandomGenerator,
):
    assert 0.0 <= roughness <= 1.0

    height, width = shape
    size = int(2**np.ceil(np.log2(max(height, width))) + 1)

    mask = np.zeros((size, size), dtype=np.float32)
    # Initialize 4 corners.
    mask[0, 0] = rng.uniform(0.0, 1.0)
    mask[0, -1] = rng.uniform(0.0, 1.0)
    mask[-1, -1] = rng.uniform(0.0, 1.0)
    mask[-1, 0] = rng.uniform(0.0, 1.0)

    step = size - 1
    iteration = 0
    while step >= 2:
        step_roughness = roughness**iteration

        squares: np.ndarray = mask[0:size:step, 0:size:step]

        square_sum_vert = squares + np.roll(squares, shift=-1, axis=0)
        square_sum_hori = squares + np.roll(squares, shift=-1, axis=1)

        # Diamond step.
        square_sum = square_sum_vert + square_sum_hori
        square_sum = square_sum[:-1, :-1]
        diamonds = ((1 - step_roughness) * square_sum / 4
                    + step_roughness * rng.uniform(0, 1, square_sum.shape))
        mask[step // 2:size:step, step // 2:size:step] = diamonds

        # Square step.
        diamond_sum_vert = diamonds + np.roll(diamonds, shift=1, axis=0)
        diamond_sum_vert = np.vstack([diamond_sum_vert, diamond_sum_vert[0]])
        square_sum0 = square_sum_hori[:, :-1] + diamond_sum_vert
        squares0 = ((1 - step_roughness) * square_sum0 / 4
                    + step_roughness * rng.uniform(0, 1, square_sum0.shape))
        mask[0:size:step, step // 2:size:step] = squares0

        diamond_sum_hori = diamonds + np.roll(diamonds, shift=1, axis=1)
        diamond_sum_hori = np.hstack([diamond_sum_hori, diamond_sum_hori[0].reshape(-1, 1)])
        square_sum1 = square_sum_vert[:-1] + diamond_sum_hori
        squares1 = ((1 - step_roughness) * square_sum1 / 4
                    + step_roughness * rng.uniform(0, 1, square_sum1.shape))
        mask[step // 2:size:step, 0:size:step] = squares1

        iteration += 1
        step = step // 2

    up = rng.integers(0, size - height + 1)
    down = up + height - 1
    left = rng.integers(0, size - width + 1)
    right = left + width - 1
    return mask[up:down + 1, left:right + 1]


@attrs.define
class FogConfig(DistortionConfig):
    roughness: float
    fog_rgb: Tuple[int, int, int] = (226, 238, 234)
    ratio_max: float = 1.0
    ratio_min: float = 0.0

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


def fog_image(
    config: FogConfig,
    state: Optional[DistortionNopState[FogConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    kind = image.kind
    image = to_rgb_image(image, kind)

    assert rng is not None
    mask = generate_diamond_square_mask(
        image.shape,
        config.roughness,
        rng,
    )
    # Equalize mask.
    mask -= mask.min()
    mask /= mask.max()
    assert config.ratio_min < config.ratio_max
    mask *= (config.ratio_max - config.ratio_min)
    mask += config.ratio_min

    mat = image.mat.astype(np.float32)

    if image.kind == ImageKind.GRAYSCALE:
        val = 0.2126 * config.fog_rgb[0] + 0.7152 * config.fog_rgb[1] + 0.0722 * config.fog_rgb[2]
        fog = np.full(image.shape, val, dtype=np.float32)
        mat = (1 - mask) * mat + mask * fog

    else:
        assert image.kind == ImageKind.RGB
        fog = np.full((*image.shape, 3), config.fog_rgb, dtype=np.float32)
        mask = np.expand_dims(mask, axis=-1)
        mat = (1 - mask) * mat + mask * fog

    mat = clip_mat_back_to_uint8(mat)
    image = attrs.evolve(image, mat=mat)

    image = to_original_image(image, kind)
    return image


fog = Distortion(
    config_cls=FogConfig,
    state_cls=DistortionNopState[FogConfig],
    func_image=fog_image,
)
