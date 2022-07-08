from typing import Optional, List, Union, Tuple
from itertools import chain

import attrs
import numpy as np
import cv2 as cv

from vkit.element import Shapable, Point, Polygon, PointList


@attrs.define
class ImageGrid:
    points_2d: List[List[Point]]

    # If set, then this is a src image grid.
    grid_size: Optional[int] = None

    cached_map_y: Optional[np.ndarray] = None
    cached_map_x: Optional[np.ndarray] = None

    _image_height: Optional[int] = None
    _image_width: Optional[int] = None

    _cached_trans_mat: Optional[List[List[Optional[np.ndarray]]]] = None
    _cached_inv_trans_mat: Optional[List[List[Optional[np.ndarray]]]] = None

    def lazy_post_init(self):
        initialized = (self._image_height is not None)
        if initialized:
            return

        assert min(point.y for point in self.flatten_points) == 0
        self._image_height = max(point.y for point in self.flatten_points) + 1

        assert min(point.x for point in self.flatten_points) == 0
        self._image_width = max(point.x for point in self.flatten_points) + 1

        if self.grid_size is not None:
            self._cached_trans_mat = [
                [None] * (self.num_cols - 1) for _ in range(self.num_rows - 1)
            ]
            self._cached_inv_trans_mat = [
                [None] * (self.num_cols - 1) for _ in range(self.num_rows - 1)
            ]
        else:
            # Avoid pickling error.
            self._cached_trans_mat = []
            self._cached_inv_trans_mat = []

    @property
    def image_height(self):
        self.lazy_post_init()
        assert self._image_height is not None
        return self._image_height

    @property
    def image_width(self):
        self.lazy_post_init()
        assert self._image_width is not None
        return self._image_width

    @property
    def cached_trans_mat(self):
        self.lazy_post_init()
        assert self._cached_trans_mat is not None
        return self._cached_trans_mat

    @property
    def cached_inv_trans_mat(self):
        self.lazy_post_init()
        assert self._cached_inv_trans_mat is not None
        return self._cached_inv_trans_mat

    @property
    def num_rows(self):
        return len(self.points_2d)

    @property
    def num_cols(self):
        return len(self.points_2d[0])

    @property
    def flatten_points(self):
        return PointList(chain.from_iterable(self.points_2d))

    @property
    def shape(self):
        return self.num_rows, self.num_cols

    def compatible_with(self, other: 'ImageGrid'):
        return self.shape == other.shape

    def generate_polygon(self, polygon_row: int, polygon_col: int):
        return Polygon(
            points=PointList([
                # Clockwise.
                self.points_2d[polygon_row][polygon_col],
                self.points_2d[polygon_row][polygon_col + 1],
                self.points_2d[polygon_row + 1][polygon_col + 1],
                self.points_2d[polygon_row + 1][polygon_col],
            ]),
        )

    def generate_polygon_row_col(self):
        for polygon_row in range(self.num_rows - 1):
            for polygon_col in range(self.num_cols - 1):
                yield polygon_row, polygon_col

    def zip_polygons(self, other: 'ImageGrid'):
        assert self.compatible_with(other)
        for polygon_row, polygon_col in self.generate_polygon_row_col():
            self_polygon = self.generate_polygon(polygon_row, polygon_col)
            other_polygon = other.generate_polygon(polygon_row, polygon_col)
            yield (polygon_row, polygon_col), self_polygon, other_polygon

    def generate_border_polygon(self):
        # Clockwise.
        points = PointList()

        for point in self.points_2d[0]:
            points.append(point)
        for row in range(1, self.num_rows):
            points.append(self.points_2d[row][-1])
        for col in reversed(range(self.num_cols - 1)):
            points.append(self.points_2d[-1][col])
        for row in reversed(range(1, self.num_rows - 1)):
            points.append(self.points_2d[row][0])

        return Polygon(points=points)

    def to_conducted_resized_image_grid(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: int,
        resized_width: int,
    ):
        new_points_2d: List[List[Point]] = []
        for points in self.points_2d:
            new_points: List[Point] = []
            for point in points:
                new_points.append(
                    point.to_conducted_resized_point(
                        shapable_or_shape=shapable_or_shape,
                        resized_height=resized_height,
                        resized_width=resized_width,
                    )
                )
            new_points_2d.append(new_points)
        return ImageGrid(points_2d=new_points_2d)

    def get_trans_mat(self, polygon_row: int, polygon_col: int, other: 'ImageGrid'):
        trans_mat: Optional[np.ndarray] = self.cached_trans_mat[polygon_row][polygon_col]

        if trans_mat is None:
            src_polygon = self.generate_polygon(polygon_row, polygon_col)
            dst_polygon = other.generate_polygon(polygon_row, polygon_col)

            trans_mat = cv.getPerspectiveTransform(
                src_polygon.to_np_array().astype(np.float32),
                dst_polygon.to_np_array().astype(np.float32),
                cv.DECOMP_SVD,
            )
            self.cached_trans_mat[polygon_row][polygon_col] = trans_mat

        assert trans_mat is not None
        return trans_mat

    def get_inv_trans_mat(self, polygon_row: int, polygon_col: int, other: 'ImageGrid'):
        inv_trans_mat: Optional[np.ndarray] = self.cached_inv_trans_mat[polygon_row][polygon_col]

        if inv_trans_mat is None:
            src_polygon = self.generate_polygon(polygon_row, polygon_col)
            dst_polygon = other.generate_polygon(polygon_row, polygon_col)

            inv_trans_mat = cv.getPerspectiveTransform(
                dst_polygon.to_np_array().astype(np.float32),
                src_polygon.to_np_array().astype(np.float32),
                cv.DECOMP_SVD,
            )
            self.cached_inv_trans_mat[polygon_row][polygon_col] = inv_trans_mat

        assert inv_trans_mat is not None
        return inv_trans_mat

    @staticmethod
    def get_np_y_x_points_within_polygon(polygon: Polygon):
        internals = polygon.to_fill_np_array_internals()
        box = internals.bounding_box
        np_mask = internals.get_np_mask()

        y, x = np_mask.nonzero()
        y += box.up
        x += box.left
        return y, x

    def generate_remap_params(self, dst_image_grid: 'ImageGrid'):
        if self.cached_map_y is not None:
            return self.cached_map_y, self.cached_map_x

        map_y = np.zeros(
            (dst_image_grid.image_height, dst_image_grid.image_width),
            dtype=np.float32,
        )
        map_x = np.zeros(
            (dst_image_grid.image_height, dst_image_grid.image_width),
            dtype=np.float32,
        )

        for (
            (polygon_row, polygon_col),
            _,
            dst_polygon,
        ) in self.zip_polygons(dst_image_grid):
            inv_trans_mat = self.get_inv_trans_mat(polygon_row, polygon_col, dst_image_grid)
            # (*, 2)
            dst_y, dst_x = ImageGrid.get_np_y_x_points_within_polygon(dst_polygon)
            # (3, *)
            dst_for_trans_all_points = np.vstack((dst_x, dst_y, np.ones_like(dst_y)))
            # (3, *)
            src_all_points = np.matmul(inv_trans_mat, dst_for_trans_all_points)

            # denominator could be zero, ignore the warning.
            denominator = src_all_points[2, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                # (2, *)
                src_all_points = src_all_points[:2, :] / denominator
            # (*, 2)
            src_all_points = src_all_points.transpose()

            zero_mask = (denominator == 0).transpose()
            if zero_mask.any():
                non_zero_mask = ~zero_mask
                src_all_points = src_all_points[non_zero_mask]
                dst_y = dst_y[non_zero_mask]
                dst_x = dst_x[non_zero_mask]

            # Split to x/y array.
            src_y = src_all_points[:, 1]
            src_x = src_all_points[:, 0]

            # Fill maps.
            map_y[dst_y, dst_x] = src_y
            map_x[dst_y, dst_x] = src_x

        self.cached_map_y = map_y
        self.cached_map_x = map_x

        return self.cached_map_y, self.cached_map_x
