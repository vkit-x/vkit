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

from vkit.element import Polygon
from vkit.mechanism.painter import Painter
from vkit.engine.char_heatmap import char_heatmap_default_engine_executor_factory
from vkit.engine.char_heatmap.default import CharHeatmapDefaultDebug

from tests.opt import write_image


def test_default():
    char_heatmap_default_engine_executor = char_heatmap_default_engine_executor_factory.create()
    polygon0 = Polygon.from_xy_pairs((
        (30, 10),
        (100, 10),
        (100, 110),
        (30, 110),
    ))
    polygon1 = Polygon.from_xy_pairs((
        (100, 10),
        (170, 10),
        (170, 110),
        (100, 110),
    ))

    polygon2 = Polygon.from_xy_pairs((
        (30, 110),
        (100, 110),
        (100, 210),
        (30, 210),
    ))
    polygon3 = Polygon.from_xy_pairs((
        (60, 110),
        (150, 110),
        (150, 210),
        (80, 210),
    ))

    char_polygons = [polygon0, polygon1, polygon2, polygon3]

    rng = default_rng(42)

    char_heatmap = char_heatmap_default_engine_executor.run(
        {
            'height': 220,
            'width': 220,
            'char_polygons': char_polygons,
            'enable_debug': True,
        },
        rng,
    )
    painter0 = Painter.create(char_heatmap.score_map)
    painter0.paint_score_map(char_heatmap.score_map)
    write_image('heatmap.png', painter0.image)

    painter1 = painter0.copy()
    painter1.paint_polygons(char_polygons, enable_polygon_points=True)
    write_image('char_polygons.png', painter1.image)

    debug = char_heatmap.debug
    assert isinstance(debug, CharHeatmapDefaultDebug)

    painter1 = painter0.copy()
    painter1.paint_mask(debug.char_overlapped_mask)
    write_image('char_overlapped_mask.png', painter1.image)

    painter1 = Painter.create(char_heatmap.score_map)
    painter1.paint_score_map(debug.char_neutralized_score_map)
    write_image('char_neutralized_score_map.png', painter1.image)

    painter1 = painter0.copy()
    painter1.paint_mask(debug.neutralized_mask)
    write_image('neutralized_mask.png', painter1.image)
