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
import cv2 as cv

from vkit.element import (
    Box,
    Polygon,
    Mask,
    Painter,
)
from tests.opt import read_image, write_image


def test_image_setitem_box():
    lenna_image = read_image('Lenna.png').to_rgb_image()
    cheems_image = read_image('Cheems.png').to_rgb_image()

    cheems_image = cheems_image.to_resized_image(resized_height=101, resized_width=101)
    box0 = Box(up=100, down=200, left=100, right=200)
    lenna_image[box0] = cheems_image

    write_image('0.jpg', lenna_image)


def test_image_setitem_polygon():
    lenna_image = read_image('Lenna.png').to_rgb_image()
    cheems_image = read_image('Cheems.png').to_rgb_image()

    cheems_image = cheems_image.to_resized_image(resized_height=101, resized_width=201)
    polygon0 = Polygon.from_xy_pairs([
        (100, 100),
        (200, 100),
        (300, 200),
        (200, 200),
    ])
    lenna_image[polygon0] = cheems_image

    write_image('0.jpg', lenna_image)


def test_image_setitem_mask():
    lenna_image = read_image('Lenna.png').to_rgb_image()
    cheems_image = read_image('Cheems.png').to_rgb_image()

    cheems_image = cheems_image.to_resized_image(resized_height=101, resized_width=101)
    edge_mask = Mask(mat=cv.Canny(cheems_image.mat, 100, 200))

    painter = Painter.create(edge_mask)
    painter.paint_mask(edge_mask)
    write_image('edge_mask.jpg', painter.image)

    np_contours, _ = cv.findContours(
        (edge_mask.mat * 255),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    assert len(np_contours) == 1
    np_contours = np.squeeze(np_contours[0], axis=1)

    polygon = Polygon.from_np_array(np_contours)

    painter = Painter.create(edge_mask)
    painter.paint_polygons([polygon])
    write_image('mask.jpg', painter.image)

    box0 = Box(up=100, down=200, left=100, right=200)
    mask = Mask.from_shapable(cheems_image).to_box_attached(box0)
    mask[polygon] = 1
    lenna_image[mask] = cheems_image
    write_image('masked_cheems.jpg', lenna_image)
