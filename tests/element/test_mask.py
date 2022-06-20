from vkit.element import FillByElementsMode, Box, Mask, MaskSetItemConfig, Painter
from tests.opt import write_image


def test_mask_setitem_box():
    mask = Mask.from_shape((400, 400))
    box0 = Box(up=100, down=200, left=100, right=200)
    mask[box0] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('box0.jpg', painter.image)

    box1 = Box(up=150, down=250, left=150, right=250)
    mask[box1] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('box1.jpg', painter.image)

    box2 = Box(up=150, down=200, left=150, right=200)
    mask[box2] = 0

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('box2.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask[(box0, box1, box2)] = 1

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('box_union.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask[(box0, box1, box2)] = MaskSetItemConfig(mode=FillByElementsMode.INTERSECT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('box_intersect.jpg', painter.image)

    mask = Mask.from_shape((400, 400))
    mask[(box0, box1, box2)] = MaskSetItemConfig(mode=FillByElementsMode.DISTINCT)

    painter = Painter.create(mask)
    painter.paint_mask(mask)
    write_image('box_distinct.jpg', painter.image)


def test_mask_setitem_polygon():
    pass
