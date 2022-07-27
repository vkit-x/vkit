from numpy.random import default_rng
from vkit.element import Painter
from vkit.engine.seal_impression import *
from tests.opt import write_image


def test_ellipse():
    engine = ellipse_seal_impression_engine_factory.create(resource=SealImpressionEngineResource())

    for rng_seed in range(10):
        # for rng_seed in (8,):
        seal_impression_layout = engine.run({'height': 400, 'width': 400}, default_rng(rng_seed))

        painter = Painter.create(seal_impression_layout.shape)
        painter.paint_mask(seal_impression_layout.background_mask)
        painter.paint_points(
            [char_slot.point_up for char_slot in seal_impression_layout.char_slots],
            color='green',
            radius=3,
        )
        painter.paint_points(
            [char_slot.point_down for char_slot in seal_impression_layout.char_slots],
            color='blue',
            radius=3,
        )
        write_image(f'layout_{rng_seed}.jpg', painter.image)
