# 噪音相关畸变

## gaussion_noise

描述：叠加高斯噪音

import:

```python
from vkit.augmentation.photometric_distortion import (
    GaussionNoiseConfig,
    gaussion_noise,
)
```

配置：

```python
@attr.define
class GaussionNoiseConfig:
    std: float
    rnd_state: Any = None
```

其中：

* `std`:  高斯噪音标准差

效果示例：

<div align="center">
    <img alt="gaussion_noise.gif" src="https://i.loli.net/2021/11/29/RLKcgotJbe3hqyf.gif" />
</div>

## poisson_noise

描述：叠加泊松噪音

import:

```python
from vkit.augmentation.photometric_distortion import (
    PoissonNoiseConfig,
    poisson_noise,
)
```

配置：

```python
@attr.define
class PoissonNoiseConfig:
    rnd_state: Any = None
```

其中：没有可以配置的选项，除了随机生成器的状态

效果示例：

<div align="center">
    <img alt="poisson_noise.gif" src="https://i.loli.net/2021/11/29/kcRW5hGMNTus9X3.gif"/>
</div>

## impulse_noise

描述：叠加脉冲噪声

import:

```python
from vkit.augmentation.photometric_distortion import (
    ImpulseNoiseConfig,
    impulse_noise,
)
```

配置：

```python
@attr.define
class ImpulseNoiseConfig:
    prob_salt: float
    prob_pepper: float
    rnd_state: Any = None
```

其中：

* `prob_salt`: 产生白色噪点（salt）的概率
* `prob_pepper`：产生黑色早点（pepper）的概率

效果示例：

<div align="center">
    <img alt="impulse_noise.gif" src="https://i.loli.net/2021/11/29/BEmACUx9ip1DeHK.gif" />
</div>

## speckle_noise

描述：叠加斑点噪声

import:

```python
from vkit.augmentation.photometric_distortion import (
    SpeckleNoiseConfig,
    speckle_noise,
)
```

配置：

```python
@attr.define
class SpeckleNoiseConfig:
    std: float
    rnd_state: Any = None
```

其中：

* `std`:  高斯斑点标准差

效果示例：

<div align="center">
    <img alt="speckle_noise.gif" src="https://i.loli.net/2021/11/29/VrQuO7GtkCzd9yE.gif" />
</div>
