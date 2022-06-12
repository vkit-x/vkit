from typing import List
from itertools import chain

from vkit.element import Point
from .type import ImageGrid
from .point_projector import PointProjector


def create_src_image_grid(height: int, width: int, grid_size: int):
    ys = list(range(0, height, grid_size))
    if ys[-1] != height - 1:
        ys.append(height - 1)

    xs = list(range(0, width, grid_size))
    if xs[-1] != width - 1:
        xs.append(width - 1)

    points_2d = []
    for y in ys:
        points = []
        for x in xs:
            points.append(Point(y=y, x=x))
        points_2d.append(points)

    return ImageGrid(
        points_2d=points_2d,
        grid_size=grid_size,
    )


def create_dst_image_grid_and_shift_amounts_and_resize_ratios(
    src_image_grid: ImageGrid,
    point_projector: PointProjector,
    resize_as_src: bool = True,
):
    dst_points_2d = []

    src_flatten_points = src_image_grid.flatten_points
    num_src_flatten_points = len(src_flatten_points)

    dst_flatten_points = point_projector.project_points(src_flatten_points)

    assert len(dst_flatten_points) == num_src_flatten_points
    dst_points_2d: List[List[Point]] = []
    for begin in range(0, num_src_flatten_points, src_image_grid.num_cols):
        dst_points_2d.append(dst_flatten_points[begin:begin + src_image_grid.num_cols])

    y_min = dst_points_2d[0][0].y
    y_max = y_min
    x_min = dst_points_2d[0][0].x
    x_max = x_min

    for point in chain.from_iterable(dst_points_2d):
        y_min = min(y_min, point.y)
        y_max = max(y_max, point.y)
        x_min = min(x_min, point.x)
        x_max = max(x_max, point.x)

    shift_amount_y = y_min
    shift_amount_x = x_min

    for point in chain.from_iterable(dst_points_2d):
        point.y -= shift_amount_y
        point.x -= shift_amount_x

    src_image_height = src_image_grid.image_height
    src_image_width = src_image_grid.image_width

    resize_ratio_y = 1.0
    resize_ratio_x = 1.0

    if resize_as_src:
        raw_dst_image_grid = ImageGrid(points_2d=dst_points_2d)
        raw_dst_image_height = raw_dst_image_grid.image_height
        raw_dst_image_width = raw_dst_image_grid.image_width
        del raw_dst_image_grid

        resize_ratio_y = (src_image_height - 1) / (raw_dst_image_height - 1)
        resize_ratio_x = (src_image_width - 1) / (raw_dst_image_width - 1)

        for point in chain.from_iterable(dst_points_2d):
            if raw_dst_image_height != src_image_height:
                point.y = round(point.y * resize_ratio_y)
            if raw_dst_image_width != src_image_width:
                point.x = round(point.x * resize_ratio_x)

    dst_image_grid = ImageGrid(points_2d=dst_points_2d)

    if resize_as_src:
        assert dst_image_grid.image_height == src_image_height
        assert dst_image_grid.image_width == src_image_width

    shift_amounts = (shift_amount_y, shift_amount_x)
    resize_ratios = (resize_ratio_y, resize_ratio_x)
    return dst_image_grid, shift_amounts, resize_ratios


def create_dst_image_grid(
    src_image_grid: ImageGrid,
    point_projector: PointProjector,
    resize_as_src: bool = True,
):
    dst_image_grid, _, _ = create_dst_image_grid_and_shift_amounts_and_resize_ratios(
        src_image_grid=src_image_grid,
        point_projector=point_projector,
        resize_as_src=resize_as_src,
    )
    return dst_image_grid
