# Interface

Import statement:

```python
from vkit.augmentation.geometric_distortion import (
    # The interface class
    GeometricDistortion,
    # The return type of distort(...)
    GeometricDistortionResult,
)
```

`GeometricDistortion.distort` interface：

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

Parameters：

* `config_or_config_generator`: A configuration instance, or a function accepting `(Tuple[int, int], np.random.RandomState)` and returning a configuration instance. Each geometric distortion strategy should be associated  with a unique configuration type, e.g. `CameraCubicCurveConfig` configuration type for  `camera_cubic_curve`
* `image`：The image to be distorted
* Parameters like `image_mask`, `image_score_map`  are optional. If provided, the same distortion operation will be applied to those objects
* `get_active_image_mask`：If set to `True`, the result should contain an  `active_image_mask` attribute, assigned by a `ImageMask` object to represent the region corresponding to the input region, so-called the active region
* `get_config`：If set to `True`, the result should contain a  `config` attribute, that is, the configuration instance guided the distortion
* `get_state`：If set to `True`, the result should contain a  `state` attribute, that is, the state instance used in distortion
* `rnd`：`numpy.random.RandomState` instance. If provided, could be used for generating configuration instance or called by any operations requiring randomness

The return type of  `GeometricDistortion.distort`：

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

Parameters: match the parameters of `GeometricDistortion.distort`
