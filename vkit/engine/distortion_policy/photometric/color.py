from typing import Tuple

import attrs
from numpy.random import RandomState

from vkit.engine import distortion
from ..type import DistortionConfigGenerator, DistortionPolicyFactory
from ..opt import LEVEL_MAX, sample_int, sample_float, sample_channels


@attrs.define
class MeanShiftConfigGeneratorConfig:
    delta_max: int = 127
    prob_negative: float = 0.5
    prob_enable_threshold: float = 0.5
    threshold_ratio_min: float = 1.0
    threshold_ratio_max: float = 1.5


class MeanShiftConfigGenerator(
    DistortionConfigGenerator[
        MeanShiftConfigGeneratorConfig,
        distortion.MeanShiftConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        delta = sample_int(
            level=self.level,
            value_min=0,
            value_max=self.config.delta_max,
            prob_negative=self.config.prob_negative,
            rnd=rnd,
        )

        channels = sample_channels(rnd)

        threshold = None
        if rnd.random() < self.config.prob_enable_threshold:
            ratio = rnd.uniform(
                self.config.threshold_ratio_min,
                self.config.threshold_ratio_max,
            )
            if delta < 0:
                threshold = round(-delta * ratio)
            else:
                threshold = round(255 - delta * ratio)

        return distortion.MeanShiftConfig(
            delta=delta,
            channels=channels,
            threshold=threshold,
        )


mean_shift_policy_factory = DistortionPolicyFactory(
    distortion.mean_shift,
    MeanShiftConfigGenerator,
)


@attrs.define
class ColorShiftConfigGeneratorConfig:
    delta_max: int = 127
    prob_negative: float = 0.5


class ColorShiftConfigGenerator(
    DistortionConfigGenerator[
        ColorShiftConfigGeneratorConfig,
        distortion.ColorShiftConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        delta = sample_int(
            level=self.level,
            value_min=0,
            value_max=self.config.delta_max,
            prob_negative=self.config.prob_negative,
            rnd=rnd,
        )

        return distortion.ColorShiftConfig(delta=delta)


color_shift_policy_factory = DistortionPolicyFactory(
    distortion.color_shift,
    ColorShiftConfigGenerator,
)


@attrs.define
class BrightnessShiftConfigGeneratorConfig:
    delta_max: int = 127
    prob_negative: float = 0.5


class BrightnessShiftConfigGenerator(
    DistortionConfigGenerator[
        BrightnessShiftConfigGeneratorConfig,
        distortion.BrightnessShiftConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        delta = sample_int(
            level=self.level,
            value_min=0,
            value_max=self.config.delta_max,
            prob_negative=self.config.prob_negative,
            rnd=rnd,
        )

        return distortion.BrightnessShiftConfig(delta=delta)


brightness_shift_policy_factory = DistortionPolicyFactory(
    distortion.brightness_shift,
    BrightnessShiftConfigGenerator,
)


@attrs.define
class StdShiftConfigGeneratorConfig:
    scale_min: float = 1.0
    scale_max: float = 2.5
    prob_reciprocal: float = 0.5


class StdShiftConfigGenerator(
    DistortionConfigGenerator[
        StdShiftConfigGeneratorConfig,
        distortion.StdShiftConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        scale = sample_float(
            level=self.level,
            value_min=self.config.scale_min,
            value_max=self.config.scale_max,
            prob_reciprocal=self.config.prob_reciprocal,
            rnd=rnd,
        )
        channels = sample_channels(rnd)

        return distortion.StdShiftConfig(
            scale=scale,
            channels=channels,
        )


std_shift_policy_factory = DistortionPolicyFactory(
    distortion.std_shift,
    StdShiftConfigGenerator,
)


@attrs.define
class BoundaryEqualizationConfigGeneratorConfig:
    pass


class BoundaryEqualizationConfigGenerator(
    DistortionConfigGenerator[
        BoundaryEqualizationConfigGeneratorConfig,
        distortion.BoundaryEqualizationConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        channels = sample_channels(rnd)

        return distortion.BoundaryEqualizationConfig(channels=channels)


boundary_equalization_policy_factory = DistortionPolicyFactory(
    distortion.boundary_equalization,
    BoundaryEqualizationConfigGenerator,
)


@attrs.define
class HistogramEqualizationConfigGeneratorConfig:
    pass


class HistogramEqualizationConfigGenerator(
    DistortionConfigGenerator[
        HistogramEqualizationConfigGeneratorConfig,
        distortion.HistogramEqualizationConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        channels = sample_channels(rnd)

        return distortion.HistogramEqualizationConfig(channels=channels)


histogram_equalization_policy_factory = DistortionPolicyFactory(
    distortion.histogram_equalization,
    HistogramEqualizationConfigGenerator,
)


@attrs.define
class ComplementConfigGeneratorConfig:
    enable_threshold_level: int = 6
    threshold_min: int = 77
    threshold_max: int = 177


class ComplementConfigGenerator(
    DistortionConfigGenerator[
        ComplementConfigGeneratorConfig,
        distortion.ComplementConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        channels = sample_channels(rnd)

        threshold = None
        enable_threshold_lte = (rnd.random() < 0.5)
        if self.level >= self.config.enable_threshold_level:
            threshold = rnd.randint(self.config.threshold_min, self.config.threshold_max + 1)

        return distortion.ComplementConfig(
            threshold=threshold,
            enable_threshold_lte=enable_threshold_lte,
            channels=channels,
        )


complement_policy_factory = DistortionPolicyFactory(
    distortion.complement,
    ComplementConfigGenerator,
)


@attrs.define
class PosterizationConfigGeneratorConfig:
    enable_threshold_level: int = 6
    threshold_min: int = 77
    threshold_max: int = 177


class PosterizationConfigGenerator(
    DistortionConfigGenerator[
        PosterizationConfigGeneratorConfig,
        distortion.PosterizationConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        # To [1, 7].
        num_bits = round(self.level / LEVEL_MAX * 7)
        channels = sample_channels(rnd)

        return distortion.PosterizationConfig(
            num_bits=num_bits,
            channels=channels,
        )


posterization_policy_factory = DistortionPolicyFactory(
    distortion.posterization,
    PosterizationConfigGenerator,
)


@attrs.define
class ColorBalanceConfigGeneratorConfig:
    ratio_min: float = 0.0
    ratio_max: float = 1.0


class ColorBalanceConfigGenerator(
    DistortionConfigGenerator[
        ColorBalanceConfigGeneratorConfig,
        distortion.ColorBalanceConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        ratio = sample_float(
            level=self.level,
            value_min=self.config.ratio_min,
            value_max=self.config.ratio_max,
            prob_reciprocal=None,
            rnd=rnd,
            inverse_level=True,
        )

        return distortion.ColorBalanceConfig(ratio=ratio)


color_balance_policy_factory = DistortionPolicyFactory(
    distortion.color_balance,
    ColorBalanceConfigGenerator,
)


@attrs.define
class ChannelPermutationConfigGeneratorConfig:
    pass


class ChannelPermutationConfigGenerator(
    DistortionConfigGenerator[
        ChannelPermutationConfigGeneratorConfig,
        distortion.ChannelPermutationConfig,
    ]
):  # yapf: disable

    def __call__(self, shape: Tuple[int, int], rnd: RandomState):
        return distortion.ChannelPermutationConfig()


channel_permutation_policy_factory = DistortionPolicyFactory(
    distortion.channel_permutation,
    ChannelPermutationConfigGenerator,
)
