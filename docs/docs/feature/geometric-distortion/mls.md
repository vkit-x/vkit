# MLS Based Distortion

## `similarity_mls`

Description: Similarity transformation as described in [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf)

Effect:

<div align="center">
    <video width="75%" height="75%" autoplay="true" muted="true" playsinline="true" loop="true" controls="ture">
        <source src="/geo/similarity_mls.mp4" type="video/mp4" />
    </video>
</div>

Import statement:

```python
from vkit.augmentation.geometric_distortion import (
    SimilarityMlsConfig,
    similarity_mls,
)
```

Configuration type:

```python
@attr.define
class SimilarityMlsConfig:
    src_handle_points: Sequence[Point]
    dst_handle_points: Sequence[Point]
    grid_size: int
    resize_as_src: bool = False
```

Parameters:

* `src_handle_points`:  The control points
* `dst_handle_points`: The deformed points
* `resize_as_src`: If set `True`, will reshape the distorted image as the original shape

