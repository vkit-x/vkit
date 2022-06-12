# 基于 MLS 的畸变

## similarity_mls

描述：参见 [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf) 文中的 similarity transformation 描述

import:

```python
from vkit.augmentation.geometric_distortion import (
    SimilarityMlsConfig,
    similarity_mls,
)
```

配置：

```python
@attr.define
class SimilarityMlsConfig:
    src_handle_points: Sequence[Point]
    dst_handle_points: Sequence[Point]
    grid_size: int
    resize_as_src: bool = False
```

其中：

* `src_handle_points` 与 `dst_handle_points` 为形变控制点
* `resize_as_src` 若设为 `True`，则强制输出图片尺寸与原图一致

效果示例：

<div align="center">
    <img alt="similarity_mls.gif" src="https://i.loli.net/2021/11/28/WjoHstxRJXmLzFT.gif" />
</div>
