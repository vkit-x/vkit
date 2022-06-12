import numpy as np
import pytest

from vkit.element import ScoreMap, Point, Image
from tests.opt import write_image


@pytest.mark.local
def test_quad_interpolation():

    for idx, points in enumerate([
        [
            Point(y=0, x=0),
            Point(y=0, x=149),
            Point(y=149, x=149),
            Point(y=149, x=0),
        ],
        [
            Point(y=0, x=50),
            Point(y=0, x=100),
            Point(y=149, x=149),
            Point(y=149, x=0),
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
            image.mat = (score_map.mat * 255).astype(np.uint8)
            write_image(f'{idx}_shift_{shift}_u.png', image)

            score_map = ScoreMap.from_quad_interpolation(
                point0=cur_points[0],
                point1=cur_points[1],
                point2=cur_points[2],
                point3=cur_points[3],
                func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1]
            )
            image = Image.from_shapable(score_map)
            image.mat = (score_map.mat * 255).astype(np.uint8)
            write_image(f'{idx}_shift_{shift}_v.png', image)

    score_map = ScoreMap.from_shape((300, 300))
    score_map.fill_by_quad_interpolation(
        point0=Point(y=0, x=0),
        point1=Point(y=0, x=149),
        point2=Point(y=149, x=149),
        point3=Point(y=149, x=0),
        func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1],
    )
    score_map.fill_by_quad_interpolation(
        point0=Point(y=200, x=50),
        point1=Point(y=250, x=150),
        point2=Point(y=280, x=150),
        point3=Point(y=280, x=50),
        func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1],
    )
    image = Image.from_shapable(score_map)
    image.mat = (score_map.mat * 255).astype(np.uint8)
    write_image('fill_by.png', image)
