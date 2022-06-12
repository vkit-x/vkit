# 基于仿射变换的畸变

## shear_hori

描述：实现横向剪切效果

import：

```python
from vkit.augmentation.geometric_distortion import (
    ShearHoriConfig,
    shear_hori,
)
```

配置：

```python
@attr.define
class ShearHoriConfig:
    # angle: int, [-90, 90], positive value for rightward direction.
    angle: int
```

其中：

* `angle`：取值范围 `(-90, 90)`，正数为向右剪切角度，负数向左

效果示例：

<div align="center">
    <img alt="shear_hori.gif" src="https://s2.loli.net/2021/12/07/ikdtLSrgaKuhANc.gif" />
</div>

## shear_vert

描述：实现纵向剪切效果

import：

```python
from vkit.augmentation.geometric_distortion import (
    ShearVertConfig,
    shear_vert,
)
```

配置：

```python
@attr.define
class ShearVertConfig:
    # angle: int, (-90, 90), positive value for downward direction.
    angle: int
```

其中：

* `angle`：取值范围 `(-90, 90)`，正数为向下剪切角度，负数向上

效果示例：

<div align="center">
    <img alt="shear_vert.gif" src="https://i.loli.net/2021/11/28/f5niNrvgWbOdRoV.gif" />
</div>

## rotate

描述：实现旋转效果

import：

```python
from vkit.augmentation.geometric_distortion import (
    RotateConfig,
    rotate,
)
```

配置：

```python
@attr.define
class RotateConfig:
    # angle: int, [0, 360], clockwise angle.
    angle: int
```

其中：

* `angle`：取值范围 `[0, 360]`，为顺时针方向角度

效果示例：

<div align="center">
    <img alt="rotate.gif" src="https://s2.loli.net/2021/12/07/5GyxcqjdVke9rJA.gif" />
</div>

## skew_hori

描述：实现水平倾斜效果

import：

```python
from vkit.augmentation.geometric_distortion import (
    SkewHoriConfig,
    skew_hori,
)
```

配置：

```python
@attr.define
class SkewHoriConfig:
    # (-1.0, 0.0], shrink the left side.
    # [0.0, 1.0), shrink the right side.
    # The larger abs(ratio), the more to shrink.
    ratio: float
```

其中：

* `ratio`：表示纵向缩减比例，取值范围 `(-1.0, 1.0)`，正数缩减右边，负数缩减左边，绝对值越大缩减的量越大，倾斜效果越明显

效果示例：

<div align="center">
    <img alt="skew_hori.gif" src="https://i.loli.net/2021/11/28/C49MQJDF2GixlXP.gif" />
</div>

## skew_vert

描述：实现垂直倾斜效果

import：

```python
from vkit.augmentation.geometric_distortion import (
    SkewVertConfig,
    skew_vert,
)
```

配置：

```python
@attr.define
class SkewVertConfig:
    # (-1.0, 0.0], shrink the up side.
    # [0.0, 1.0), shrink the down side.
    # The larger abs(ratio), the more to shrink.
    ratio: float
```

其中：

* `ratio`：表示横向缩减比例，取值范围 `(-1.0, 1.0)`，正数缩减下边，负数缩减上边，绝对值越大缩减的量越大，倾斜效果越明显

效果示例：

<div align="center">
    <img alt="skew_vert.gif" src="https://i.loli.net/2021/11/28/V9cOmJZuRLXlk8r.gif" />
</div>

