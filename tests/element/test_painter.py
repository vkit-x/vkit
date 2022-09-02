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
from vkit.element import *
from tests.opt import read_image, write_image


def test_painter():
    painter = Painter.create((500, 500))
    painter.paint_points(
        points=PointList.from_xy_pairs([
            (0, 0),
            (50, 100),
            (100, 100),
            (499, 499),
        ])
    )
    write_image('paint_points.jpg', painter.image)

    painter = Painter.create((500, 500))
    painter.paint_points(
        points=PointList.from_xy_pairs([
            (0, 0),
            (50, 100),
            (100, 100),
            (499, 499),
        ]),
        radius=10,
        enable_index=True,
    )
    write_image('paint_points_r3_index.jpg', painter.image)

    painter = Painter(read_image('Lenna.png'))
    painter.paint_lines(
        lines=[
            Line.from_xy_pairs([(0, 0), (100, 0)]),
            Line.from_xy_pairs([(50, 50), (300, 300)]),
        ]
    )
    write_image('paint_lines.jpg', painter.image)

    painter = Painter.create((500, 500))
    painter.paint_lines(
        lines=[
            Line.from_xy_pairs([(0, 0), (100, 0)]),
            Line.from_xy_pairs([(50, 50), (300, 300)]),
        ],
        thickness=10,
        enable_arrow=True,
    )
    write_image('paint_lines_t10_arrow.jpg', painter.image)

    painter = Painter.create((500, 500))
    painter.paint_lines(
        lines=[
            Line.from_xy_pairs([(0, 0), (100, 0)]),
            Line.from_xy_pairs([(50, 50), (300, 300)]),
        ],
        thickness=10,
        enable_arrow=True,
        arrow_length_ratio=0.5,
    )
    write_image('paint_lines_t10_arrow_0.5.jpg', painter.image)

    painter = Painter.create((500, 500))
    painter.paint_boxes(
        boxes=[
            Box(up=0, down=99, left=0, right=99),
            Box(up=300, down=399, left=300, right=399),
        ]
    )
    write_image('paint_boxes.jpg', painter.image)

    painter = Painter.create((500, 500))
    painter.paint_boxes(
        boxes=[
            Box(up=0, down=99, left=0, right=99),
            Box(up=300, down=399, left=300, right=399),
        ],
        border_thickness=2,
    )
    write_image('paint_boxes_no_filled.jpg', painter.image)

    painter = Painter(read_image('Lenna.png'))
    painter.paint_boxes(
        boxes=[
            Box(up=0, down=99, left=0, right=99),
            Box(up=300, down=399, left=300, right=399),
        ],
        border_thickness=2,
    )
    write_image('paint_boxes_no_filled2.jpg', painter.image)

    painter = Painter(read_image('Lenna.png'))
    painter.paint_polygons(
        polygons=[Polygon.from_xy_pairs([
            (0, 0),
            (100, 100),
            (100, 300),
            (50, 200),
        ])]
    )
    write_image('paint_polygons.jpg', painter.image)

    painter = Painter(read_image('Lenna.png'))
    mask = Mask.from_shapable(painter.image)
    mask.fill_by_polygons([Polygon.from_xy_pairs([
        (0, 0),
        (100, 100),
        (100, 300),
        (50, 200),
    ])])
    painter.paint_mask(mask)
    write_image('paint_mask.jpg', painter.image)

    painter = Painter(read_image('Lenna.png'))
    score_map = ScoreMap.from_shapable(painter.image)
    score_map.fill_by_polygons([
        Polygon.from_xy_pairs([
            (0, 0),
            (100, 100),
            (100, 300),
            (50, 200),
        ])
    ])
    painter.paint_score_map(score_map)
    write_image('paint_score_map.jpg', painter.image)
