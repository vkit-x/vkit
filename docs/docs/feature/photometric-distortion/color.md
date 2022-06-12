# Colorspace Related Distortion

## `mean_shift`

Description: Shift the mean of each channel, aka the brightness

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/mean_shift.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    MeanShiftConfig,
    mean_shift,
)
```

Configuration type:

```python
@attr.define
class MeanShiftConfig:
    delta: int
```

Parameters:

* `delta`: The value for addition. Overflow/underflow issue of `uint8` is resolved through clipping

## `std_shift`

Description: Shift (adjust) the standard deviation of each channel, aka the contrast

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/std_shift.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    StdShiftConfig,
    std_shift,
)
```

Configuration type:

```python
@attr.define
class StdShiftConfig:
    scale: float
```

Parameters:

* `scale`: The value for multiplication. Overflow/underflow issue of `uint8` is resolved through clipping

## `channel_permutate`

Description: Permutate the order of channels


Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/channel_permutate.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    ChannelPermutateConfig,
    channel_permutate,
)
```

Configuration type:

```python
@attr.define
class ChannelPermutateConfig:
    rnd_state: Any = None
```

Parameters:

* `rnd_state`: Optional. If provided, should be the same type as `numpy.random.RandomState.get_state()`. `rnd_state` is used to initialize `numpy.random.RandomState` to control the randomness


## `hue_shift`

Description: Shift the mean of hue channel. Note that the input `Image` should have `HSV` mode

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/hue_shift.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    HueShiftConfig,
    hue_shift,
)
```

Configuration type:

```python
@attr.define
class HueShiftConfig:
    delta: int
```

Parameters:

* `delta`: The value for hue addition. Overflow/underflow issue of `uint8` is resolved through modulo operation

## `saturation_shift`

Description: Shift the mean of saturation channel. Note that the input `Image` should have `HSV` mode

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/saturation_shift.mp4" type="video/mp4" />
    </video>
</div>


Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    SaturationShiftConfig,
    saturation_shift,
)
```

Configuration type:

```python
@attr.define
class SaturationShiftConfig:
    delta: int
```

Parameters:

* `delta`: The value for saturation addition. Overflow/underflow issue of `uint8` is resolved through clipping
