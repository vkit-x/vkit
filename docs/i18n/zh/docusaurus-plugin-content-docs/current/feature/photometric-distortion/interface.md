# 接口说明

Import 示例:

```python
from vkit.augmentation.photometric_distortion import (
    PhotometricDistortion,
)
```

`PhotometricDistortion.distort_image` 接口：

```python
def distort_image(
    self,
    config_or_config_generator: Union[T_CONFIG,
                                      Callable[[Tuple[int, int], np.random.RandomState],
                                               T_CONFIG]],
    image: Image,
    rnd: Optional[np.random.RandomState] = None,
) -> Image:
    ...
```

其中：

* `config_or_config_generator`：传入光度畸变配置，或者传入一个生成配置的函数。每种光度畸变的操作，都有对应的独立配置类型，如 `mean_shift` 对应 `MeanShiftConfig`

* `image`：需要进行光度畸变的图片
* `rnd`：`numpy.random.RandomState` 实例，用于生成配置或者其他需要随机行为的操作

与几何畸变不同的是，光度畸变并不会改变图片中元素的位置，所以并没有对标注类型（如 `ImageMask`）的处理接口。`distort_image` 的函数名也比较明确，即光度畸变的处理对象是图片，返回被处理过的新图片
