import numpy as np

from vkit.element import FillByElementsMode, Box, Polygon, Mask, MaskSetItemConfig, Painter
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
    mask[(box0, box1, box2)] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('union.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask[(box0, box1, box2)] = MaskSetItemConfig(mode=FillByElementsMode.INTERSECT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('intersect.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask[(box0, box1, box2)] = MaskSetItemConfig(mode=FillByElementsMode.DISTINCT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('distinct.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    boxed_mask = Mask.from_shape(box0.shape)
    rnd = np.random.RandomState(0)
    boxed_mask.mat[rnd.random(size=boxed_mask.shape) > 0.5] = 1
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
    mask[(polygon0, polygon1)] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('union.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask[(polygon0, polygon1)] = MaskSetItemConfig(mode=FillByElementsMode.INTERSECT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('intersect.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask[(polygon0, polygon1)] = MaskSetItemConfig(mode=FillByElementsMode.DISTINCT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('distinct.jpg', painter.image)
