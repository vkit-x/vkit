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
from numpy.random import default_rng

from vkit.element import FillByElementsMode, Box, Polygon, Mask, Painter
from tests.opt import write_image


def test_mask_setitem_box():
    mask = Mask.from_shape((400, 400))
    box0 = Box(up=100, down=200, left=100, right=200)
    mask[box0] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('0.jpg', painter.image)

    box1 = Box(up=150, down=250, left=150, right=250)
    mask[box1] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('1.jpg', painter.image)

    box2 = Box(up=150, down=200, left=150, right=200)
    mask[box2] = 0

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('2.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask.fill_by_boxes((box0, box1, box2), 1)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('union.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask.fill_by_boxes((box0, box1, box2), mode=FillByElementsMode.INTERSECT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('intersect.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask.fill_by_boxes((box0, box1, box2), mode=FillByElementsMode.DISTINCT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('distinct.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    boxed_mask = Mask.from_shape(box0.shape)
    rng = default_rng(0)
    boxed_mask.mat[rng.random(size=boxed_mask.shape) > 0.5] = 1
    mask[box0] = boxed_mask

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('boxed_mask.jpg', painter.image)


def test_mask_setitem_polygon():
    mask = Mask.from_shape((400, 400))
    polygon0 = Polygon.from_xy_pairs([
        (100, 100),
        (200, 100),
        (300, 200),
        (200, 200),
    ])
    mask[polygon0] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('0.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    polygon1 = Polygon.from_xy_pairs([
        (200, 100),
        (300, 100),
        (200, 200),
        (100, 200),
    ])
    mask[polygon1] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('1.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask.fill_by_polygons((polygon0, polygon1))

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('union.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask.fill_by_polygons((polygon0, polygon1), mode=FillByElementsMode.INTERSECT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('intersect.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask.fill_by_polygons((polygon0, polygon1), mode=FillByElementsMode.DISTINCT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('distinct.jpg', painter.image)
