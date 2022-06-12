# Labeled Data Type

## Point

Import statement:

```python
from vkit.label.type import Point
```

`Point` represents a point on 2D plane：

```python
@attr.define
class Point:
    y: int
    x: int
```

Methods available in `Point`:

### `self.clone`

Parameters: None


Returns a copy of the object

### `self.to_xy_pair`

Parameters: None

Returns a tuple as `(x, y)`

### `self.to_clipped_point`

Parameters: `image: Image`

Generates a new `Point`, ensure no positional overflow/underflow would take place via clip operation

### `self.to_resized_point`

Parameters: `image: Image, resized_height: int, resized_width: int`

Generates new `Point` based on resized target image's height and width. `image` represents the original image, `resized_height` and `resized_width` is the desired resized image's height and width respectively

## PointList

Import statement:

```python
from vkit.label.type import PointList
```

`PointList` is used to represent a list of `Point`：

```python
class PointList(List[Point]):
    ...
```

Methods available in `PointList`：

### `PointList.from_np_array`

Parameters: `np_points: np.ndarray`

Converts a numpy array with shape `(*, 2)` (`[(x, y), ...]`) into a `PointList` object

### `PointList.from_xy_pairs`

Parameters: `xy_pairs: Iterable[Tuple[int, int]]`

Converts an `Iterable[Tuple[int, int]]` into a `PointList` object

### `PointList.from_flatten_xy_pairs`

Parameters: `flatten_xy_pairs: Sequence[int]`

Similar to `PointList.from_xy_pairs`, however accepts the input in the format of `[x0, y0, x1, y1, ...]`

### `PointList.from_point`

Parameters: `point: Point`

Returns a `PointList` which contains the specified `point` object alone

### `self.clone`

Parameters: None

Returns a copy of the object

### `self.to_xy_pairs`

Parameters: None

Converts the `PointList` into `List[Tuple[int, int]]`, aka the reverse operation of `PointList.from_xy_pairs`

### `self.to_np_array`

Parameters: None

Converts the `PointList` into a numpy array, aka the reverse operation of `PointList.from_np_array`

### `self.to_clipped_points`

Parameters: `image: Image`

Generates a new `PointList` object, ensures no positional overflow/underflow via clip operation

### `self.to_resized_points`

Parameters: `image: Image, resized_height: int, resized_width: int`

Generates a new `PointList` based on resized target image's height and width. `image` represents the original image, `resized_height` and `resized_width` is the desired resized image's height and width respectively

## Box

Import statement:

```python
from vkit.label.type import Box
```

`Box` is used to represent rectangular labeled area which is horizontal and vertical

```python
@attr.define
class Box:
    up: int
    down: int
    left: int
    right: int
```

Explanation:

* `up` and `left` represents the coordinate of the top left corner
* `up` and `right` represents the coordinate of the top right corner
* `down` and `left` represents the coordinate of the bottom left corner
* `down` and `right` represents the coordinate of the bottom right corner
* Box is therefore a rectangular enclosed by the 4 points mentioned above

Attributes of `Box`:

* `height`: type `int`
* `width`: type `int`
* `shape`: (height, width), type `Tuple[int, int]`

Methods available for `Box`:

### `self.clone`

Parameters: None

Creates a copy of the `Box`

### `self.to_clipped_box`

Parameters: `image: Image`

Generates a new `Box` object, ensures no positional overflow/underflow via clip operation

### `self.extract_image`

Parameters: `image: Image`

Extracts part of a `image` surrounded by a `Box` object, returns a `Image` object. Note that this method will not generate a new numpy array, therefore explicit `clone` is required

## Polygon

Import statement:

```python
from vkit.label.type import Polygon
```

`Polygon` represents a polygon shaped area

```python
@attr.define
class Polygon:
    points: PointList
```

Methods available for `Polygon`:

### `Polygon.from_np_array`

Parameters: `np_points: np.ndarray`

Calls `PointList.from_np_array` to generate `self.points`

### `Polygon.from_xy_pairs`

Parameters: `xy_pairs`

Calls `PointList.from_xy_pairs` to generate `self.points`

### `Polygon.from_flatten_xy_pairs`

Parameters: `xy_pairs: Sequence[int]`

Calls `PointList.from_flatten_xy_pairs` to generate`self.points`

### `self.to_xy_pairs`, `self.to_np_array` and `self.to_clipped_points`

Will all call methods defined in `PointList` with the same name and output in the same type

### `self.to_clipped_polygon`

Parameters: None

Returns `Polygon` while `self.to_clipped_points()` returns `PointList`

### `self.to_bounding_box_with_np_points`

Parameters: `shift_np_points: bool = False`

Returns `Tuple[Box, np.ndarray]`, which is the bounding `Box` object and `self.points` converted to a numpy array. If `shift_np_points` was set to `True`, the point that was closest to origin will be the new origin (aka. shift to `(0, 0)`)

### `self.to_bounding_box`

Parameters: None

Returns the `Box` explained in `self.to_bounding_box_with_np_points`

### `self.to_resized_polygon`

Parameters: `image: Image, resized_height: int, resized_width: int`

Generates a new `Polygon` based on resized target image's height and width. `image` represents the original image, `resized_height` and `resized_width` is the desired resized image's height and width respectively

### `self.clone`

Parameters: None

Returns a copy of the object

## TextPolygon

Import statement:

```python
from vkit.label.type import TextPolygon
```

`TextPolygon` represents a polygon area tagged with text label:

```python
@attr.define
class TextPolygon:
    text: str
    polygon: Polygon
    meta: Optional[Dict[str, Any]] = None
```

Explanation:

* `text`: must not be empty or `None`
* `meta`: Optional, can be used to keep other metadata

Methods available for `TextPolygon`:

### `self.to_resized_text_polygon`

Parameters: `image: Image, resized_height: int, resized_width: int`

Generates a new `TextPolygon` based on resized target image's height and width. `image` represents the original image, `resized_height` and `resized_width` is the desired resized image's height and width respectively

## ImageMask

Import statement:

```python
from vkit.label.type import ImageMask
```

`ImageMask` represents a mask label:

```python
@attr.define
class ImageMask:
    mat: np.ndarray
```

Explanation:

* `mat` fulfills `ndim = 2` and `dtype = np.uint8`

Attributes of `ImageMask`:

* `height`: type `int`
* `width`: type `int`
* `shape`: (height, width), type `Tuple[int, int]`

Methods available for `ImageMask`:

### `ImageMask.from_shape`

Parameters: `height: int, width: int`

Initialize `ImageMask` from `shape`, while all elements in `mat` will be initialized to `0`

### `ImageMask.from_shape_and_polygons`

Parameters: `height: int, width: int, polygons: Iterable[Polygon], mode: ImageMaskPolygonsMergeMode = ImageMaskPolygonsMergeMode.UNION`

Initialize `ImageMask` from `shape` and one or more `Polygon`.

The default `mode == ImageMaskPolygonsMergeMode.UNION` sets the areas covered by any of the `Polygon` to `1`; If `mode == ImageMaskPolygonsMergeMode.DISTINCT`, only sets non-overlapping areas to `1`; Similarly, if `mode == ImageMaskPolygonsMergeMode.INTERSECTION`, only sets areas where overlap exist to `1`

### `ImageMask.from_image_and_polygons`

Parameters: `image: Image, polygons: Iterable[Polygon], mode: ImageMaskPolygonsMergeMode = ImageMaskPolygonsMergeMode.UNION`

Similar to `ImageMask.from_shape_and_polygons`, however `image.shape` will be used instead

### `self.to_resized_image_mask`

Parameters: `height: int, width: int, cv_resize_interpolation: int = cv.INTER_NEAREST_EXACT`

resize the height and width of the mask

### `self.clone`

Parameters: None

Returns a copy of the object

## ImageScoreMap

Import statement:

```python
from vkit.label.type import ImageScoreMap
```

`ImageScoreMap` represents a score map：

```python
@attr.define
class ImageScoreMap:
    mat: np.ndarray
```

Explanation:

* `mat` fulfills `ndim = 2` and `dtype = np.float32`

Attributes of `ImageScoreMap`:

* `height`: type `int`
* `width`: type `int`
* `shape`: (height, width), type `Tuple[int, int]`

Methods available in `ImageScoreMap`：

### `ImageScoreMap.from_image_mask`

Parameters: `image_mask: ImageMask`

Converts from a `ImageMask` to a `ImageScoreMap`

### `ImageScoreMap.from_shape_and_polygon_value_pairs`

Parameters: `height: int, width: int, polygon_value_pairs: Iterable[Tuple[Polygon, float]]`

Initialize a `ImageScoreMap` with the provided height and width, the float value in the `Tuple[Polygon, float]` will be used as the score

### `ImageScoreMap.from_image_and_polygon_value_pairs`

Parameters: `image: Image, polygon_value_pairs: Iterable[Tuple[Polygon, float]]`

Similar to `ImageScoreMap.from_shape_and_polygon_value_pairs`, however `image.shape` will be used instead for the height and width
