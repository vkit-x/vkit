# Affine Transformation Based Distortion

## `shear_hori`

Description: Implement the horizontal shearing effects

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/geo/shear_hori.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.geometric_distortion import (
    ShearHoriConfig,
    shear_hori,
)
```

Configuration type:

```python
@attr.define
class ShearHoriConfig:
    angle: int
```

Parameters:

* `angle`: The shearing angle in range `(-90, 90)`. Positive value for rightward shearing

## `shear_vert`

Description: Implement the vertical shearing effects

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/geo/shear_vert.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.geometric_distortion import (
    ShearVertConfig,
    shear_vert,
)
```

Configuration type:

```python
@attr.define
class ShearVertConfig:
    angle: int
```

Parameters:

* `angle`: The shearing angle in range `(-90, 90)`. Positive value for downward shearing

## `rotate`

Description: Implement the rotation effects

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/geo/rotate.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.geometric_distortion import (
    RotateConfig,
    rotate,
)
```

Configuration type:

```python
@attr.define
class RotateConfig:
    angle: int
```

Parameters:

* `angle`: The clockwise rotation angle in range `[0, 360]`

## `skew_hori`

Description: Implement horizontal skewing effects

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/geo/skew_hori.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.geometric_distortion import (
    SkewHoriConfig,
    skew_hori,
)
```

Configuration type:

```python
@attr.define
class SkewHoriConfig:
    ratio: float
```

Parameters:

* `ratio`: The ratio to shew, with value in range `(-1.0, 1.0)`. Positive value for skewing to the right, while negative value for skewing to the left. The degree of skewness increases with the absolute value of `ratio`

## `skew_vert`

Description: Implement vertical skewing effects

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/geo/skew_vert.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.geometric_distortion import (
    SkewVertConfig,
    skew_vert,
)
```

Configuration type:

```python
@attr.define
class SkewVertConfig:
    ratio: float
```

Parameters:

* `ratio`: The ratio to shew, with value in range `(-1.0, 1.0)`. Positive value for skewing to the top, while negative value for skewing to the bottom. The degree of skewness increases with the absolute value of `ratio`
