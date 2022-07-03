from typing import Any, Optional, Mapping

import attrs
import numpy as np
from numpy.random import Generator as RandomGenerator

from vkit.element import Image
from ..interface import DistortionConfig, DistortionNopState, Distortion
from .opt import extract_mat_from_image, clip_mat_back_to_uint8


@attrs.define
class GaussionNoiseConfig(DistortionConfig):
    std: float

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


def gaussion_noise_image(
    config: GaussionNoiseConfig,
    state: Optional[DistortionNopState[GaussionNoiseConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    assert rng
    mat = extract_mat_from_image(image, np.int16)
    noise = np.round(rng.normal(0, config.std, mat.shape)).astype(np.int16)
    mat = clip_mat_back_to_uint8(mat + noise)
    return Image(mat=mat)


gaussion_noise = Distortion(
    config_cls=GaussionNoiseConfig,
    state_cls=DistortionNopState[GaussionNoiseConfig],
    func_image=gaussion_noise_image,
)


@attrs.define
class PoissonNoiseConfig(DistortionConfig):
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


def poisson_noise_image(
    config: PoissonNoiseConfig,
    state: Optional[DistortionNopState[PoissonNoiseConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    assert rng
    mat = rng.poisson(extract_mat_from_image(image, np.float32))
    mat = clip_mat_back_to_uint8(mat)
    return Image(mat=mat)


poisson_noise = Distortion(
    config_cls=PoissonNoiseConfig,
    state_cls=DistortionNopState[PoissonNoiseConfig],
    func_image=poisson_noise_image,
)


@attrs.define
class ImpulseNoiseConfig(DistortionConfig):
    prob_salt: float
    prob_pepper: float

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


def impulse_noise_image(
    config: ImpulseNoiseConfig,
    state: Optional[DistortionNopState[ImpulseNoiseConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    assert rng

    # https://www.programmersought.com/article/3363136769/
    prob_presv = 1 - config.prob_salt - config.prob_pepper
    assert prob_presv >= 0.0

    mask = rng.choice(
        (0, 1, 2),
        size=image.shape,
        p=[prob_presv, config.prob_salt, config.prob_pepper],
    )

    mat = image.mat.copy()
    # Salt.
    mat[mask == 1] = 255
    # Pepper.
    mat[mask == 2] = 0

    return Image(mat=mat)


impulse_noise = Distortion(
    config_cls=ImpulseNoiseConfig,
    state_cls=DistortionNopState[ImpulseNoiseConfig],
    func_image=impulse_noise_image,
)


@attrs.define
class SpeckleNoiseConfig(DistortionConfig):
    std: float

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


def speckle_noise_image(
    config: SpeckleNoiseConfig,
    state: Optional[DistortionNopState[SpeckleNoiseConfig]],
    image: Image,
    rng: Optional[RandomGenerator],
):
    assert rng
    mat = extract_mat_from_image(image, np.float32)
    noise = rng.normal(0, config.std, mat.shape)
    mat = clip_mat_back_to_uint8(mat + mat * noise)
    return Image(mat=mat)


speckle_noise = Distortion(
    config_cls=SpeckleNoiseConfig,
    state_cls=DistortionNopState[SpeckleNoiseConfig],
    func_image=speckle_noise_image,
)
