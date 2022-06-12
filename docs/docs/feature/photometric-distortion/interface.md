# Interface

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    PhotometricDistortion,
)
```

`PhotometricDistortion.distort_image` interface:

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

Parameters:

* `config_or_config_generator`: A configuration instance, or a function accepting `(Tuple[int, int], np.random.RandomState)` and returning a configuration instance. Each photometric distortion strategy should be associated  with a unique configuration type, e.g. `MeanShiftConfig` configuration type for  `mean_shift`
* `image`: The image to be distorted
* `rnd`: `numpy.random.RandomState`: `numpy.random.RandomState` instance. If provided, could be used for generating configuration instance or called by any operations requiring randomness

Unlike geometric distortion, photometric distortion doesn't change the position of element. Hence, there's no interface designed to accept labeled data types.
