# TODO
# def debug():
#     from numpy.random import default_rng
#     from vkit.utility import get_data_folder
#     import iolite as io
#     fd = io.folder(get_data_folder(__file__), touch=True)

#     from .type import FontCollection
#     import os.path
#     font_collection = FontCollection.from_folder(
#         os.path.expandvars('$VKIT_PRIVATE_DATA/vkit_font/font_collection')
#     )
#     name_to_font_meta = {font_meta.name: font_meta for font_meta in font_collection.font_metas}
#     # font_meta = name_to_font_meta['方正书宋简体']
#     # font_meta = name_to_font_meta['STXihei']
#     font_meta = name_to_font_meta['NotoSansSC']

#     from vkit.engine.interface import EngineExecutorFactory

#     config = FontEngineRunConfig(
#         height=12,
#         width=640,
#         # height=640,
#         # width=32,
#         chars=list('我可以吞下玻璃，且不伤害到自-己??'),
#         font_variant=font_meta.get_font_variant(1),
#         # chars=list('this is good.'),
#         # chars=[],
#         # style=FontEngineRenderTextLineStyle(
#         #     glyph_sequence=FontEngineRenderTextLineStyleGlyphSequence.VERT_DEFAULT
#         # ),
#     )
#     rng = default_rng(42)
#     # result = EngineFactory(FontFreetypeDefaultEngine).create().run(config, rng)
#     result = EngineExecutorFactory(FontFreetypeMonochromeEngine).create().run(config, rng)
#     assert result is not None
#     result.image.to_file(fd / 'image.png')

#     from vkit.element import Painter

#     painter = Painter.create(result.mask)
#     painter.paint_mask(result.mask, alpha=1.0)
#     painter.to_file(fd / 'mask.png')

#     painter = Painter.create(result.image)
#     painter.paint_char_boxes(result.char_boxes)
#     painter.to_file(fd / 'char_boxes.png')

# def debug_ratio():
#     from numpy.random import default_rng
#     from .type import FontCollection
#     import os.path
#     font_collection = FontCollection.from_folder(
#         os.path.expandvars('$VKIT_PRIVATE_DATA/vkit_font/font_collection')
#     )
#     name_to_font_meta = {font_meta.name: font_meta for font_meta in font_collection.font_metas}
#     font_meta = name_to_font_meta['方正书宋简体']

#     from vkit.engine.interface import EngineExecutorFactory

#     font_variant = font_meta.get_font_variant(0)
#     shapes = []
#     engine = EngineExecutorFactory(FontFreetypeDefaultEngine).create()
#     rng = default_rng(0)
#     from tqdm import tqdm
#     for char in tqdm(font_meta.chars):
#         config = FontEngineRunConfig(
#             height=32,
#             width=64,
#             chars=[char],
#             font_variant=font_variant,
#         )
#         result = engine.run(config, rng)
#         assert result
#         assert len(result.char_boxes) == 1
#         char_box = result.char_boxes[0]
#         shapes.append((char_box.height, char_box.width))

#     ratios = [height / width for height, width in shapes]
#     print('mean', np.mean(ratios))
#     print('median', np.median(ratios))
