# 基于相机模型的畸变

## camera_cubic_curve

描述：实现基于相机模型的与三次函数的 3D 曲面效果，参见 [Page dewarping](https://mzucker.github.io/2016/08/15/page-dewarping.html) 文中描述

import:

```python
from vkit.augmentation.geometric_distortion import (
    CameraModelConfig,
    CameraCubicCurveConfig,
    camera_cubic_curve,
)
```

配置：

```python
@attr.define
class CameraModelConfig:
    rotation_unit_vec: Sequence[float]
    rotation_theta: float
    principal_point: Optional[Sequence[float]] = None
    focal_length: Optional[float] = None
    camera_distance: Optional[float] = None


@attr.define
class CameraCubicCurveConfig:
    curve_alpha: float
    curve_beta: float
    # Clockwise, [0, 180]
    curve_direction: float
    curve_scale: float
    camera_model_config: CameraModelConfig
    grid_size: int
```

其中：

* `CameraModelConfig` 是相机模型的配置（下同）
  * `rotation_unit_vec`：旋转单位向量，即旋转指向方向，具体见 [cv.Rodrigues](https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac) 里的解释
  * `rotation_theta`：旋转的角度，区间 `[-180, 180]`。角度为正数，表示的方向与右手法则旋转方向一致
  * `principal_point`：可选。相机成像光学轴（optical axis）与图片的相交点，使用原图的坐标表示。如果不提供，默认会设定为图片的中心点
  * `focal_length`：可选。相机成像光学轴的焦距。如果不提供，会采用图像长宽中较大值作为焦距
  * `camera_distance`：可选。指定相机坐标原点到 `principal_point` 的距离。如果不提供，会基于图片与成像屏幕相切的策略决定此距离

* `CameraCubicCurveConfig` 控制如何生成曲面
  * `curve_alpha`：投影左端点的斜率
  * `curve_beta`：投影右端点的斜率
  * `curve_direction`：投影线的方向，区间 `[0, 180]`。图片会按照这个方生成曲面，例如角度为 `0` 时曲面的“起伏”是横向的，`90` 时为纵向。基于投影位置，会生成 Z 轴的偏移量
  * `curve_scale`：控制 Z 轴的偏移量的放大倍数，建议设为 `1.0`
  * `grid_size`：网格的大小，下同。网格越小，几何畸变效果越好，性能越差


效果示例：

<div align="center">
    <img alt="camera_cubic_curve.gif" src="https://i.loli.net/2021/11/25/B7Rpz46u5axO1sf.gif" />
</div>

其中（下同）：

* 左上：形变后图片，`Image`
* 右上：形变后多边形，`Polygon`
* 左中：形变后图像平面网格，`ImageGrid`
* 右中：图像 `active_image_mask` 蒙板，`ImageMask`
* 左下：形变后图像蒙版，`ImageMask`
* 右下：形变后评分图，`ImageScoreMap`

## camera_plane_line_fold

描述：实现基于相机模型与基准线的翻折效果，参见 [DocUNet: Document Image Unwarping via A Stacked U-Net](https://www3.cs.stonybrook.edu/~cvl/docunet.html) 文中描述

import:

```python
from vkit.augmentation.geometric_distortion import (
    CameraModelConfig,
    CameraPlaneLineFoldConfig,
    camera_plane_line_fold,
)
```

配置：

```python
@attr.define
class CameraPlaneLineFoldConfig:
    fold_point: Tuple[float, float]
    # Clockwise, [0, 180]
    fold_direction: float
    fold_perturb_vec: Tuple[float, float, float]
    fold_alpha: float
    camera_model_config: CameraModelConfig
    grid_size: int
```

其中：

* `fold_point`  与  `fold_direction`  决定基准线。 `fold_point`  设为原图的某个点，`fold_direction` 为从该点出发的基准线角度，顺时针区值区间 `[0, 180]`
* `fold_perturb_vec`：为三维扰动向量。图中的点与基准线越接近，扰动越强，即 `p + w * fold_perturb_vec`
* `fold_alpha`： 控制 `w = fold_alpha / (fold_alpha + d)`，`d` 为点到翻折线的归一化距离。`fold_alpha` 的取值越靠近 `0`，翻折效果越强。推荐取值 `0.5`

效果示例：

<div align="center">
    <img alt="camera_plane_line_fold.gif" src="https://i.loli.net/2021/11/25/FLicMRwuA1tynrg.gif" />
</div>

## camera_plane_line_curve

描述：实现基于相机模型的与基准线的曲面效果，参见 [DocUNet: Document Image Unwarping via A Stacked U-Net](https://www3.cs.stonybrook.edu/~cvl/docunet.html) 文中描述

import:

```python
from vkit.augmentation.geometric_distortion import (
    CameraModelConfig,
    CameraPlaneLineCurveConfig,
    camera_plane_line_curve,
)
```

配置：

```python
@attr.define
class CameraPlaneLineCurveConfig:
    curve_point: Tuple[float, float]
    # Clockwise, [0, 180]
    curve_direction: float
    curve_perturb_vec: Tuple[float, float, float]
    curve_alpha: float
    camera_model_config: CameraModelConfig
    grid_size: int

```

其中：

* `curve_point`  与  `curve_direction`  决定基准线，同 `CameraPlaneLineFoldConfig`
* `curve_perturb_vec`：为三维扰动向量。同 `CameraPlaneLineFoldConfig`
* `curve_alpha`： 控制 `w = 1 - d^curve_alpha`，`d` 为点到基准线的归一化距离。`curve_alpha`  越小，越接近翻折的效果。推荐取值 `2.0`

效果示例：

<div align="center">
    <img alt="camera_plane_line_curve.gif" src="https://i.loli.net/2021/11/26/xcCPAUbZDflO3wj.gif" />
</div>
