from typing import cast, Any, Optional, Mapping, Sequence

import attrs
import numpy as np
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.element import Image, ImageKind
from ..interface import DistortionConfig, DistortionNopState, Distortion
from .opt import (
    extract_mat_from_image,
    clip_mat_back_to_uint8,
    handle_out_of_bound_and_dtype,
    generate_new_image,
    OutOfBoundBehavior,
)


def _mean_shift(
    image: Image,
    channels: Optional[Sequence[int]],
    delta: int,
    threshold: Optional[int],
    oob_behavior: OutOfBoundBehavior,
):
    if delta == 0:
        return image

    mat = extract_mat_from_image(image, np.int16, channels)

    if threshold is None:
        mat += delta
    else:
        if delta > 0:
            mask = (mat <= threshold)
        else:
            assert delta < 0
            mask = (threshold <= mat)
        mat[mask] += delta

    mat = handle_out_of_bound_and_dtype(mat, oob_behavior)
    return generate_new_image(image, mat, channels)


@attrs.define
class MeanShiftConfig(DistortionConfig):
    delta: int
    threshold: Optional[int] = None
    channels: Optional[Sequence[int]] = None
    oob_behavior: OutOfBoundBehavior = OutOfBoundBehavior.CLIP


def mean_shift_image(
    config: MeanShiftConfig,
    state: Optional[DistortionNopState[MeanShiftConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    return _mean_shift(
        image=image,
        channels=config.channels,
        delta=config.delta,
        threshold=config.threshold,
        oob_behavior=config.oob_behavior,
    )


mean_shift = Distortion(
    config_cls=MeanShiftConfig,
    state_cls=DistortionNopState[MeanShiftConfig],
    func_image=mean_shift_image,
)


@attrs.define
class ColorShiftConfig(DistortionConfig):
    delta: int


def color_shift_image(
    config: ColorShiftConfig,
    state: Optional[DistortionNopState[ColorShiftConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    kind = image.kind
    if kind not in (ImageKind.HSV, ImageKind.HSL):
        # To HSV.
        image = image.to_hsv_image()

    image = _mean_shift(
        image=image,
        # Operate the hue channel.
        channels=[0],
        delta=config.delta,
        threshold=None,
        oob_behavior=OutOfBoundBehavior.CYCLE,
    )

    if kind not in (ImageKind.HSV, ImageKind.HSL):
        image = image.to_target_kind_image(kind)

    return image


color_shift = Distortion(
    config_cls=ColorShiftConfig,
    state_cls=DistortionNopState[ColorShiftConfig],
    func_image=color_shift_image,
)


@attrs.define
class BrightnessShiftConfig(DistortionConfig):
    delta: int
    intermediate_kind: ImageKind = ImageKind.HSL


def brightness_shift_image(
    config: BrightnessShiftConfig,
    state: Optional[DistortionNopState[BrightnessShiftConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    kind = image.kind
    if kind not in (ImageKind.HSV, ImageKind.HSL):
        assert config.intermediate_kind in (ImageKind.HSV, ImageKind.HSL)
        image = image.to_target_kind_image(config.intermediate_kind)

    image = _mean_shift(
        image=image,
        # Operate the lighting channel.
        channels=[2],
        delta=config.delta,
        threshold=None,
        oob_behavior=OutOfBoundBehavior.CLIP,
    )

    if kind not in (ImageKind.HSV, ImageKind.HSL):
        image = image.to_target_kind_image(kind)

    return image


brightness_shift = Distortion(
    config_cls=BrightnessShiftConfig,
    state_cls=DistortionNopState[BrightnessShiftConfig],
    func_image=brightness_shift_image,
)


def _std_shift(
    image: Image,
    channels: Optional[Sequence[int]],
    scale: float,
    oob_behavior: OutOfBoundBehavior,
):
    mat = extract_mat_from_image(image, np.float32, channels)

    assert scale > 0
    if mat.ndim == 2:
        mean = np.mean(mat)
    elif mat.ndim == 3:
        mean = np.mean(mat.reshape(-1, mat.shape[-1]), axis=0)
    else:
        raise NotImplementedError()
    mat = mat * scale - mean * (scale - 1)

    mat = handle_out_of_bound_and_dtype(mat, oob_behavior)
    return generate_new_image(image, mat, channels)


@attrs.define
class StdShiftConfig(DistortionConfig):
    scale: float
    channels: Optional[Sequence[int]] = None


def std_shift_image(
    config: StdShiftConfig,
    state: Optional[DistortionNopState[StdShiftConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    return _std_shift(
        image,
        config.channels,
        config.scale,
        OutOfBoundBehavior.CLIP,
    )


std_shift = Distortion(
    config_cls=StdShiftConfig,
    state_cls=DistortionNopState[StdShiftConfig],
    func_image=std_shift_image,
)


@attrs.define
class BoundaryEqualizationConfig(DistortionConfig):
    channels: Optional[Sequence[int]] = None


def boundary_equalization_image(
    config: BoundaryEqualizationConfig,
    state: Optional[DistortionNopState[BoundaryEqualizationConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    mat = extract_mat_from_image(image, np.float32, config.channels)

    # Equalize each channel to [0, 255].
    if mat.ndim == 2:
        delta: np.ndarray = mat.max() - mat.min()
        if delta == 0.0:
            return image

        mat -= mat.min()
        mat *= 255.0 / delta

    elif mat.ndim == 3:
        flatten_mat = mat.reshape(-1, mat.shape[-1])
        val_min = flatten_mat.min(axis=0)
        val_max = flatten_mat.max(axis=0)
        delta = val_max - val_min

        mask = (delta > 0)
        if not mask.any():
            return image

        num_channels = mask.sum()
        masked_min = mat[:, :, mask].reshape(-1, num_channels).min(axis=0)
        mat[:, :, mask] -= masked_min
        mat[:, :, mask] *= 255.0 / delta[mask]

    else:
        raise NotImplementedError()

    mat = handle_out_of_bound_and_dtype(mat, OutOfBoundBehavior.CLIP)
    return generate_new_image(image, mat, config.channels)


boundary_equalization = Distortion(
    config_cls=BoundaryEqualizationConfig,
    state_cls=DistortionNopState[BoundaryEqualizationConfig],
    func_image=boundary_equalization_image,
)


@attrs.define
class HistogramEqualizationConfig(DistortionConfig):
    channels: Optional[Sequence[int]] = None


def histogram_equalization_image(
    config: HistogramEqualizationConfig,
    state: Optional[DistortionNopState[HistogramEqualizationConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    mat = extract_mat_from_image(image, np.uint8, config.channels)

    if mat.ndim == 2:
        channel_mats: Sequence[np.ndarray] = [mat]
    elif mat.ndim == 3:
        channel_mats: Sequence[np.ndarray] = np.dsplit(mat, mat.shape[-1])
    else:
        raise NotImplementedError()

    new_mats = [cv.equalizeHist(channel_mat) for channel_mat in channel_mats]

    if mat.ndim == 2:
        return attrs.evolve(image, mat=new_mats[0])
    elif mat.ndim == 3:
        return generate_new_image(image, np.dstack(new_mats), config.channels)
    else:
        raise NotImplementedError()


histogram_equalization = Distortion(
    config_cls=HistogramEqualizationConfig,
    state_cls=DistortionNopState[HistogramEqualizationConfig],
    func_image=histogram_equalization_image,
)


@attrs.define
class ComplementConfig(DistortionConfig):
    threshold: Optional[int] = None
    enable_threshold_lte: bool = False
    channels: Optional[Sequence[int]] = None


def complement_image(
    config: ComplementConfig,
    state: Optional[DistortionNopState[ComplementConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    mat = extract_mat_from_image(image, np.uint8, config.channels)

    if config.threshold is None:
        mat = 255 - mat
    else:
        assert 0 <= config.threshold <= 255
        if not config.enable_threshold_lte:
            mask = (config.threshold <= mat)
        else:
            mask = (mat <= config.threshold)
        mat[mask] = 255 - mat[mask]

    return generate_new_image(image, mat, config.channels)


complement = Distortion(
    config_cls=ComplementConfig,
    state_cls=DistortionNopState[ComplementConfig],
    func_image=complement_image,
)


@attrs.define
class PosterizationConfig(DistortionConfig):
    num_bits: int
    channels: Optional[Sequence[int]] = None


def posterization_image(
    config: PosterizationConfig,
    state: Optional[DistortionNopState[PosterizationConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    assert 0 <= config.num_bits < 8

    if config.num_bits == 0:
        return image

    mat = extract_mat_from_image(image, np.uint8, config.channels)
    # Clear lower n bits.
    mat = np.bitwise_and(mat, (0xFF >> config.num_bits) << config.num_bits)
    return generate_new_image(image, mat, config.channels)


posterization = Distortion(
    config_cls=PosterizationConfig,
    state_cls=DistortionNopState[PosterizationConfig],
    func_image=posterization_image,
)


@attrs.define
class ColorBalanceConfig(DistortionConfig):
    ratio: float


def color_balance_image(
    config: ColorBalanceConfig,
    state: Optional[DistortionNopState[ColorBalanceConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    if image.kind == ImageKind.GRAYSCALE:
        return image

    grayscale_like_image = image.to_grayscale_image().to_target_kind_image(image.kind)
    grayscale_like_mat = grayscale_like_image.mat.astype(np.float32)
    mat = image.mat.astype(np.float32)

    if image.kind in (ImageKind.HSV, ImageKind.HSL):
        channels = cast(Sequence[int], [1, 2])
        grayscale_like_mat = grayscale_like_mat[:, :, channels]
        mat = mat[:, :, channels]

    assert 0.0 <= config.ratio <= 1.0
    mat = (1 - config.ratio) * grayscale_like_mat + config.ratio * mat
    mat = clip_mat_back_to_uint8(mat)

    if image.kind in (ImageKind.HSV, ImageKind.HSL):
        return generate_new_image(image, mat, [1, 2])
    else:
        return attrs.evolve(image, mat=mat)


color_balance = Distortion(
    config_cls=ColorBalanceConfig,
    state_cls=DistortionNopState[ColorBalanceConfig],
    func_image=color_balance_image,
)


@attrs.define
class ChannelPermutationConfig(DistortionConfig):
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


def channel_permutation_image(
    config: ChannelPermutationConfig,
    state: Optional[DistortionNopState[ChannelPermutationConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    assert rng
    indices = rng.permutation(image.num_channels)
    mat = image.mat[:, :, indices]
    return attrs.evolve(image, mat=mat)


channel_permutation = Distortion(
    config_cls=ChannelPermutationConfig,
    state_cls=DistortionNopState[ChannelPermutationConfig],
    func_image=channel_permutation_image,
)
