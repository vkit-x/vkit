# 接口说明

Import 示例:

```python
from vkit.augmentation.geometric_distortion import (
    # 接口类型
    GeometricDistortion,
    # distort(...) 返回类型
    GeometricDistortionResult,
    # 具体的几何畸变实现
    ...
)
```

`GeometricDistortion.distort` 接口：

```python
def distort(
    self,
    config_or_config_generator: Union[T_CONFIG,
                                      Callable[[Tuple[int, int], np.random.RandomState],
                                               T_CONFIG]],
    image: Image,
    image_mask: Optional[ImageMask] = None,
    image_score_map: Optional[ImageScoreMap] = None,
    point: Optional[Point] = None,
    points: Optional[PointList] = None,
    polygon: Optional[Polygon] = None,
    polygons: Optional[Iterable[Polygon]] = None,
    get_active_image_mask: bool = False,
    get_config: bool = False,
    get_state: bool = False,
    rnd: Optional[np.random.RandomState] = None,
) -> GeometricDistortionResult:
    ...
```

其中：

* `config_or_config_generator`：传入几何畸变配置，或者传入一个生成配置的函数。每种几何畸变的操作，都有对应的独立配置类型，如 `camera_cubic_curve` 对应 `CameraCubicCurveConfig`
* `image`：需要进行几何畸变的图片
* `image_mask`, `image_score_map` 等皆为可选项，会对传入对象执行与 `image` 一致的几何畸变
* `get_active_image_mask`：如果设置，会在结果中返回 `active_image_mask` 蒙板，用于表示变换后属于原图的激活区域
* `get_config`：如果设置，会在结果中返回配置实例
* `get_state`：如果设置，会在结果中返回状态实例
* `rnd`：`numpy.random.RandomState` 实例，用于生成配置或者其他需要随机行为的操作

`GeometricDistortion.distort` 接口返回类型：

```python
@attr.define
class GeometricDistortionResult:
    image: Image
    image_mask: Optional[ImageMask] = None
    image_score_map: Optional[ImageScoreMap] = None
    active_image_mask: Optional[ImageMask] = None
    point: Optional[Point] = None
    points: Optional[PointList] = None
    polygon: Optional[Polygon] = None
    polygons: Optional[Sequence[Polygon]] = None
    config: Optional[Any] = None
    state: Optional[Any] = None
```

其中，返回的字段对应传入参数。
