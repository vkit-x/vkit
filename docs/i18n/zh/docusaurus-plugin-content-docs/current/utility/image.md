# 图像类型

## ImageKind

import：

```python
from vkit.image.type import ImageKind
```

`ImageKind` 用于标记 `Image` 的图片类型：

```python
class ImageKind(Enum):
    RGB = auto()
    RGB_GCN = auto()
    RGBA = auto()
    HSV = auto()
    HSV_GCN = auto()
    GRAYSCALE = auto()
    GRAYSCALE_GCN = auto()
    NONE = auto()
```

其中：

* `*_GCN`： 表示对应类型的的 GCN（Global Contrast Normalization）后的结果类型
* `RGB`： 关联 `mat.ndim = 3`，`mat.dtype = np.uint8`
* `RGB_GCN`： 关联 `mat.ndim = 3`，`mat.dtype = np.float32`
* `RGBA`： 关联 `mat.ndim = 4`，`mat.dtype = np.uint8`
* `HSV`： 关联 `mat.ndim = 3`，`mat.dtype = np.uint8`
* `HSV_GCN`： 关联 `mat.ndim = 3`，`mat.dtype = np.float32`
* `GRAYSCALE`： 关联 `mat.ndim = 2`，`mat.dtype = np.uint8`
* `GRAYSCALE_GCN`： 关联 `mat.ndim = 2`，`mat.dtype = np.float32`
* `NONE`： 仅在 `Image` 初始化过程使用，在 `Image`  没有显式传入 `kind` 时，vkit 会根据 `mat` 的 `ndim` 与 `dtype` 自动推导出 `kind`

## Image

import：

```python
from vkit.image.type import Image
```

`Image` 是 vkit 封装的图像数据类型，支持 I/O、归一化、缩放等操作。`Image` 的数据字段如下：

```python
@attr.define
class Image:
    mat: np.ndarray
    kind: ImageKind = ImageKind.NONE
```

其中：

* `mat`：是一个 numpy array，其 `ndim` 与 `dtype` 与 `kind`  关联，见上方
* `kind`：用于标记 `mat`

`Image` 的属性：

* `height`：高，类型 `int`
* `width`：宽，类型 `int`
* `shape`：（高，宽），类型 `Tuple[int, int]`
* `num_channels`： 通道数，类型 `int`。如果类型属于 `GRAYSCALE`，`GRAYSCALE_GCN`，返回 `0`

`Image` 的 I/O 方法：

* `ImageKind.from_file(path: PathType, disable_exif_orientation: bool = False)`：直接从图片文件路径实例化 `Image`。默认 `disable_exif_orientation = False` 时，会从图片文件中解析 EXIF 元数据，执行相关旋转操作
* `self.to_file(path: PathType, disable_to_rgb_image: bool = False)`：将 `Image` 输出到文件。默认 `disable_to_rgb_image: bool = False` 时，会自动将图片转为 RGB 格式保存
* `ImageKind.from_pil_image(pil_image: Image.Image)`：从 `PIL.Image` 实例化 `Image`
* `self.to_pil_image()`：将 `Image` 转换为 `PIL.Image`

`Image` 的转换方法：

* `self.clone()`：复制 `Image`
* `self.to_grayscale_image()`：将 `Image` 转为 `GRAYSCALE` 类型。如果 `self` 本身已经是 `GRAYSCALE` 类型，会返回一个 `clone` 实例
* `self.to_rgb_image()`：将 `Image` 转为 `RGB` 类型。如果 `self` 本身已经是 `RGB` 类型，会返回一个 `clone` 实例
* `self.to_rgba_image()`：将 `Image` 转为 `RGBA` 类型。如果 `self` 本身已经是 `RGBA` 类型，会返回一个 `clone` 实例
* `self.to_hsv_image()`：将 `Image` 转为 `HSV` 类型。如果 `self` 本身已经是 `HSV` 类型，会返回一个 `clone` 实例
* `self.to_gcn_image(lamb=0, eps=1E-8, scale=1.0)`，对图片执行 GCN 操作，详情见 [此文](https://cedar.buffalo.edu/~srihari/CSE676/12.2%20Computer%20Vision.pdf)
* `self.to_non_gcn_image()`：将图片转换为对应的非 GCN 类型，如 `RGB_GCN -> RGB`
* `self.to_resized_image(self, height: int, width: int, cv_resize_interpolation: int = cv.INTER_CUBIC)`：缩放图片的高度与宽度
