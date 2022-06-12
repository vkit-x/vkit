# Noise Related Distortion

## `gaussion_noise`

Description: Superimpose gaussion noise

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/gaussion_noise.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    GaussionNoiseConfig,
    gaussion_noise,
)
```

Configuration type:

```python
@attr.define
class GaussionNoiseConfig:
    std: float
    rnd_state: Any = None
```

Parameters:

* `std`:  The standard deviation of gaussion noise
* `rnd_state`: Same as the description in [Colorspace Related Distortion](color.md)

## `poisson_noise`

Description: Superimpose poisson noise

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/poisson_noise.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    PoissonNoiseConfig,
    poisson_noise,
)
```

Configuration type:

```python
@attr.define
class PoissonNoiseConfig:
    rnd_state: Any = None
```

Parameters:

* `rnd_state`: Same as the description in [Colorspace Related Distortion](color.md)

## `impulse_noise`

Description: Superimpose impulse noise

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/impulse_noise.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    ImpulseNoiseConfig,
    impulse_noise,
)
```

Configuration type:

```python
@attr.define
class ImpulseNoiseConfig:
    prob_salt: float
    prob_pepper: float
    rnd_state: Any = None
```

Parameters:

* `prob_salt`: The probability to generate white noise pixel, aka the salt
* `prob_pepper`: The probability to generate black noise pixel, aka the pepper
* `rnd_state`: Same as the description in [Colorspace Related Distortion](color.md)

## `speckle_noise`

Description: Superimpose speckle noise

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/pho/speckle_noise.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.photometric_distortion import (
    SpeckleNoiseConfig,
    speckle_noise,
)
```

Configuration type:

```python
@attr.define
class SpeckleNoiseConfig:
    std: float
    rnd_state: Any = None
```

Parameters:

* `std`: The standard deviation of speckle noise
* `rnd_state`: Same as the description in [Colorspace Related Distortion](color.md)

