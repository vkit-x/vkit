# 色彩空间相关畸变

## mean_shift

描述：调整每个通道的均值。即通俗说法中的亮度调整

import:

```python
from vkit.augmentation.photometric_distortion import (
    MeanShiftConfig,
    mean_shift,
)
```

配置：

```python
@attr.define
class MeanShiftConfig:
    delta: int
```

其中：

* `delta`: 相加用的值。已经考虑   `uint8`  overflow/underflow 的问题

效果示例：

<div align="center">
    <img alt="brightness_shift.gif" src="https://i.loli.net/2021/11/28/QZAsdRmTYJcjG1K.gif" />
</div>

## std_shift

描述：调整每个通道的标准差，同时保持通道的均值。即通俗说法中的对比度调整

import:

```python
from vkit.augmentation.photometric_distortion import (
    StdShiftConfig,
    std_shift,
)
```

配置：

```python
@attr.define
class StdShiftConfig:
    scale: float
```

其中：

* `scale`: 相乘用的值。已经考虑   `uint8`  overflow/underflow 的问题

效果示例：

<div align="center">
    <img alt=".gif" src="https://i.loli.net/2021/11/28/zaW1KCeLxgs4Yop.gif" />
</div>

## channel_permutate

描述：随机重组通道的顺序

import:

```python
from vkit.augmentation.photometric_distortion import (
    ChannelPermutateConfig,
    channel_permutate,
)
```

配置：

```python
@attr.define
class ChannelPermutateConfig:
    rnd_state: Any = None
```

其中：

* `rnd_state`: 可选，类型与  `numpy.random.RandomState.get_state()` 的返回值一致，用于初始化 `numpy.random.RandomState`。默认情况会随机初始化

效果示例：

<div align="center">
    <img alt="channel_permutate.gif" src="https://i.loli.net/2021/11/28/ySkFD7YXbtul2Ji.gif" />
</div>

## hue_shift

描述：调整 HSV 色彩空间中的色调（hue）值。注意传入的图片的模式需要是 HSV

import:

```python
from vkit.augmentation.photometric_distortion import (
    HueShiftConfig,
    hue_shift,
)
```

配置：

```python
@attr.define
class HueShiftConfig:
    delta: int
```

其中：

* `delta`: 色调相加的值。会通过取 mod 的模式处理 overflow/underflow 问题

效果示例：

<div align="center">
    <img alt="hue_shift.gif" src="https://i.loli.net/2021/11/29/JSTem4yocrB1WUs.gif" />
</div>

## saturation_shift

描述：调整 HSV 色彩空间中的饱和度（saturation）值。注意传入的图片的模式需要是 HSV

import:

```python
from vkit.augmentation.photometric_distortion import (
    SaturationShiftConfig,
    saturation_shift,
)
```

配置：

```python
@attr.define
class SaturationShiftConfig:
    delta: int
```

其中：

* `delta`: 饱和度相加的值

效果示例：

<div align="center">
    <img alt="saturation_shift.gif" src="https://i.loli.net/2021/11/29/ON8jEdIbmWX1VFo.gif" />
</div>

