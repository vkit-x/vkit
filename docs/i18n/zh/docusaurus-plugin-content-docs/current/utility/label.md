# 标注类型

## Point

import：

```python
from vkit.label.type import Point
```

`Point` 用于表示二维平面上的点：

```python
@attr.define
class Point:
    y: int
    x: int
```

`Point` 的方法：

* `self.clone()` ：返回拷贝
* `self.to_xy_pair()`：返回 `(x, y)`
* `self.to_clipped_point(image: Image)`：生成新的 `Point` ，通过 clip 操作保证不会出现位置的 overflow/underflow
* `self.to_resized_point(image: Image, resized_height: int, resized_width: int)`：基于目标缩放图片的高度与宽度，生成新的 `Point`。`image` 是缩放前图片，`resized_height` 与 `resized_width` 是缩放后的图片高度与宽度

## PointList

import：

```python
from vkit.label.type import PointList
```

`PointList` 用于表示 `Point` 数组：

```python
class PointList(List[Point]):
    ...
```

`PointList` 的方法：

* `PointList.from_np_array(np_points: np.ndarray)`：将 numpy array `(*, 2)` （`[(x, y), ...]`）转换为 `PointList`
* `PointList.from_xy_pairs(xy_pairs: Iterable[Tuple[int, int]])`： 将 `Iterable[Tuple[int, int]]` 转换为  `PointList`
* `PointList.from_flatten_xy_pairs(flatten_xy_pairs: Sequence[int])`：类似 `PointList.from_xy_pairs`，但输入形式为 `[x0, y0, x1, y1, ...]`
* `PointList.from_point(point: Point)`：返回包含 `point` 为唯一元素的 `PointList`
* `self.clone()`：返回拷贝
* `self.to_xy_pairs()`：转换为 `List[Tuple[int, int]]` 格式，即 `PointList.from_xy_pairs` 的逆过程
* `self.to_np_array()`：转换为 numpy array，即 `PointList.from_np_array` 的逆过程
* `self.to_clipped_points(image: Image)`：生成新的 `PointList` ，通过 clip 操作保证不会出现位置的 overflow/underflow
* `self.to_resized_points(image: Image, resized_height: int, resized_width: int)`：基于目标缩放图片的高度与宽度，生成新的 `PointList`。`image` 是缩放前图片，`resized_height` 与 `resized_width` 是缩放后的图片高度与宽度

## Box

import：

```python
from vkit.label.type import Box
```

`Box` 用于表示横平竖直的矩形标注区域：

```python
@attr.define
class Box:
    up: int
    down: int
    left: int
    right: int
```

其中：

* `down` 与 `right` 的表示是闭区间端点

`Box` 的属性：

* `height`：高度，类型 `int`
* `width`：宽度，类型 `int`
* `shape`：（高度，宽度），类型 `Tuple[int, int]`

`Box`  的方法：

* `self.clone()`：返回拷贝
* `self.to_clipped_box(image: Image)`：生成新的 `Box` ，通过 clip 操作保证不会出现位置的 overflow/underflow
* `self.extract_image(image: Image)`：从 `image` 中抽取 `Box`  划定的区域，返回一个新的 `Image`。需要注意，这个操作不会产生一个新的 numpy array，如有需要得显式地调用 `clone`

## Polygon

import：

```python
from vkit.label.type import Polygon
```

`Polygon` 用于表示多边形标注区域：

```python
@attr.define
class Polygon:
    points: PointList
```

`Polygon` 的方法：

* `Polygon.from_np_array(np_points: np.ndarray)`：调用 `PointList.from_np_array` 生成 `self.points`
* `Polygon.from_xy_pairs(xy_pairs)`：调用 `PointList.from_xy_pairs` 生成 `self.points`
* `Polygon.from_flatten_xy_pairs(xy_pairs: Sequence[int])`：调用 `PointList.from_flatten_xy_pairs` 生成 `self.points`
* `self.to_xy_pairs()`、`self.to_np_array()`、`self.to_clipped_points` 皆在内部调用 `PointList` 同名方法，生成同样类型的输出
* `self.to_clipped_polygon()` 与 `self.to_clipped_points()`，区别在于返回 `Polygon`
* `self.to_bounding_box_with_np_points(shift_np_points: bool = False)`：返回 `Tuple[Box, np.ndarray]` ，即外接矩形 `Box` 与转为 numpy array 格式的  `self.points`。如果将 `shift_np_points` 设为 `True`，则会将 numpy array 中离原点最近的点设为原点（shift 至 `(0, 0)`）
* `self.to_bounding_box()`：返回 `self.to_bounding_box_with_np_points` 中的 `Box`
* `self.to_resized_polygon(image: Image, resized_height: int, resized_width: int)`：基于目标缩放图片的高度与宽度，生成新的 `Polygon`。`image` 是缩放前图片，`resized_height` 与 `resized_width` 是缩放后的图片高度与宽度
* `self.clone()`：返回拷贝

## TextPolygon

import：

```python
from vkit.label.type import TextPolygon
```

`TextPolygon` 用于表示带文本标注的多边形标注区域：

```python
@attr.define
class TextPolygon:
    text: str
    polygon: Polygon
    meta: Optional[Dict[str, Any]] = None
```

其中：

* `text`：必须不为空
* `meta`：可选。用于存储额外字段

`TextPolygon` 的方法：

* `self.to_resized_text_polygon(image: Image, resized_height: int, resized_width: int)`：基于目标缩放图片的高度与宽度，生成新的 `TextPolygon`。`image` 是缩放前图片，`resized_height` 与 `resized_width` 是缩放后的图片高度与宽度

## ImageMask

import：

```python
from vkit.label.type import ImageMask
```

`ImageMask` 用于表示蒙板（mask）标注：

```python
@attr.define
class ImageMask:
    mat: np.ndarray
```

其中：

* `mat`：`ndim = 2` 且 `dtype = np.uint8`

`ImageMask` 的属性：

* `height`：高，类型 `int`
* `width`：宽，类型 `int`
* `shape`：（高，宽），类型 `Tuple[int, int]`

`ImageMask` 的方法：

* `ImageMask.from_shape(height: int, width: int)`：从形状初始化 `ImageMask`，`mat` 初始化为 `0`
* `ImageMask.from_shape_and_polygons(height: int, width: int, polygons: Iterable[Polygon], mode: ImageMaskPolygonsMergeMode = ImageMaskPolygonsMergeMode.UNION)`：从形状与多边形初始化 `ImageMask`。默认 `mode == ImageMaskPolygonsMergeMode.UNION` 时，将所有多边形区域设为 `1`；如果  `mode == ImageMaskPolygonsMergeMode.DISTINCT`，只将非相交区域设为 `1`；如果  `mode == ImageMaskPolygonsMergeMode.INTERSECTION`，只将重合区域设为 `1`
* `ImageMask.from_image_and_polygons(image: Image, polygons: Iterable[Polygon], mode: ImageMaskPolygonsMergeMode = ImageMaskPolygonsMergeMode.UNION)`：与 `ImageMask.from_shape_and_polygons` 类似，只不过会采用 `image.shape`
* `self.to_resized_image_mask(height: int, width: int, cv_resize_interpolation: int = cv.INTER_NEAREST_EXACT)`：缩放蒙板的高度与宽度
* `self.clone()`：返回拷贝

## ImageScoreMap

import：

```python
from vkit.label.type import ImageScoreMap
```

`ImageScoreMap` 用于表示评分图：

```python
@attr.define
class ImageScoreMap:
    mat: np.ndarray
```

其中：

* `mat`：`ndim = 2` 且 `dtype = np.float32`

`ImageScoreMap` 的属性：

* `height`：高，类型 `int`
* `width`：宽，类型 `int`
* `shape`：（高，宽），类型 `Tuple[int, int]`

`ImageScoreMap` 的方法：

* `ImageScoreMap.from_image_mask(image_mask: ImageMask)`：从 `ImageMask` 转换生成
* `ImageScoreMap.from_shape_and_polygon_value_pairs(height: int, width: int, polygon_value_pairs: Iterable[Tuple[Polygon, float]])`：初始化（高，宽）的评分图，图中的多边形使用对应的评分赋值
* `ImageScoreMap.from_image_and_polygon_value_pairs(image: Image, polygon_value_pairs: Iterable[Tuple[Polygon, float]])`：与 `ImageScoreMap.from_shape_and_polygon_value_pairs` 类似，只不过会采用 `image.shape`
