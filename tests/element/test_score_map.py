# Copyright 2022 vkit-x Administrator. All Rights Reserved.
#
# This project (vkit-x/vkit) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
import numpy as np
from numpy.random import default_rng
import pytest

from vkit.element import (
    FillByElementsMode,
    Point,
    Box,
    Polygon,
    Mask,
    ScoreMap,
    Image,
    Painter,
)
from tests.opt import write_image


@pytest.mark.local
def test_quad_interpolation():

    for idx, points in enumerate([
        [
            Point.create(y=0, x=0),
            Point.create(y=0, x=149),
            Point.create(y=149, x=149),
            Point.create(y=149, x=0),
        ],
        [
            Point.create(y=0, x=50),
            Point.create(y=0, x=100),
            Point.create(y=149, x=149),
            Point.create(y=149, x=0),
        ],
    ]):
        for shift in range(4):
            cur_points = points[shift:] + points[:shift]
            # print(idx, shift, cur_points)

            score_map = ScoreMap.from_quad_interpolation(
                point0=cur_points[0],
                point1=cur_points[1],
                point2=cur_points[2],
                point3=cur_points[3],
                func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 0]
            )
            image = Image.from_shapable(score_map)
            image.assign_mat((score_map.mat * 255).astype(np.uint8))
            write_image(f'{idx}_shift_{shift}_u.png', image)

            score_map = ScoreMap.from_quad_interpolation(
                point0=cur_points[0],
                point1=cur_points[1],
                point2=cur_points[2],
                point3=cur_points[3],
                func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1]
            )
            image = Image.from_shapable(score_map)
            image.assign_mat((score_map.mat * 255).astype(np.uint8))
            write_image(f'{idx}_shift_{shift}_v.png', image)

    score_map = ScoreMap.from_shape((300, 300))
    score_map.fill_by_quad_interpolation(
        point0=Point.create(y=0, x=0),
        point1=Point.create(y=0, x=149),
        point2=Point.create(y=149, x=149),
        point3=Point.create(y=149, x=0),
        func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1],
    )
    score_map.fill_by_quad_interpolation(
        point0=Point.create(y=200, x=50),
        point1=Point.create(y=250, x=150),
        point2=Point.create(y=280, x=150),
        point3=Point.create(y=280, x=50),
        func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1],
    )
    image = Image.from_shapable(score_map)
    image.assign_mat((score_map.mat * 255).astype(np.uint8))
    write_image('fill_by.png', image)


def test_score_map_setitem_box():
    score_map = ScoreMap.from_shape((400, 400))
    box0 = Box(up=100, down=200, left=100, right=200)
    score_map[box0] = 1.0

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('0.jpg', painter.image)

    box1 = Box(up=150, down=250, left=150, right=250)
    score_map[box1] = 0.5

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('1.jpg', painter.image)

    score_map = ScoreMap.from_shape((400, 400))
    score_map.fill_by_boxes((box0, box1))

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('union.jpg', painter.image)

    score_map = ScoreMap.from_shape((400, 400))
    score_map.fill_by_boxes((box0, box1), mode=FillByElementsMode.INTERSECT)

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('intersect.jpg', painter.image)

    score_map = ScoreMap.from_shape((400, 400))
    score_map.fill_by_boxes((box0, box1), mode=FillByElementsMode.DISTINCT)

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('distinct.jpg', painter.image)

    box2 = Box(up=200, down=300, left=200, right=300)
    score_map = ScoreMap.from_shape((400, 400))
    score_map.fill_by_box_value_pairs(
        zip((box0, box1, box2), [1.0, 0.75, 0.5]), keep_max_value=True
    )

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('keep_max.jpg', painter.image)

    score_map = ScoreMap.from_shape((400, 400), value=1.0)
    score_map.fill_by_box_value_pairs(
        zip((box0, box1, box2), [1.0, 0.75, 0.5]), keep_min_value=True
    )

    mask = Mask.from_shapable(score_map)
    mask.fill_by_boxes((box0, box1, box2))
    mask.to_inverted_mask().fill_score_map(score_map, 0.0)

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('keep_min.jpg', painter.image)


def test_score_map_setitem_polygon():
    score_map = ScoreMap.from_shape((400, 400))
    polygon0 = Polygon.from_xy_pairs([
        (100, 100),
        (200, 100),
        (300, 200),
        (200, 200),
    ])
    score_map[polygon0] = 1

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('0.jpg', painter.image)

    polygon1 = Polygon.from_xy_pairs([
        (200, 100),
        (300, 100),
        (200, 200),
        (100, 200),
    ])
    score_map[polygon1] = 1

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('1.jpg', painter.image)

    score_map = ScoreMap.from_shape((400, 400))
    score_map.fill_by_polygon_value_pairs(zip((polygon0, polygon1), (0.75, 0.5)))
    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('2.jpg', painter.image)


def test_score_map_setitem_mask():
    score_map = ScoreMap.from_shape((400, 400))

    box0 = Box(up=100, down=200, left=100, right=200)
    boxed_mask = Mask.from_shape(box0.shape).to_box_attached(box0)
    rng = default_rng(0)
    with boxed_mask.writable_context:
        boxed_mask.mat[rng.random(size=boxed_mask.shape) > 0.5] = 1
    score_map[boxed_mask] = 1.0

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('conducted.jpg', painter.image)

    score_map = ScoreMap.from_shape((400, 400))
    boxed_mask = Mask.from_shapable(score_map)
    rng = default_rng(0)
    with boxed_mask.writable_context:
        boxed_mask.mat[rng.random(size=boxed_mask.shape) > 0.5] = 1
    score_map[boxed_mask] = 1.0

    painter = Painter.create(score_map)
    painter.paint_score_map(score_map)
    write_image('full.jpg', painter.image)
