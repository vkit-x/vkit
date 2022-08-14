# flake8: noqa
# TODO: improve
from numpy.random import default_rng
import numpy as np
import cattrs
import pytest

from vkit.element import Image, Mask, ScoreMap, Painter
from vkit.pipeline import (
    Pipeline,
    bypass_post_processor_factory,
    page_shape_step_factory,
    page_background_step_factory,
    page_layout_step_factory,
    PageLayout,
    page_text_line_step_factory,
    PageTextLineCollection,
    page_image_step_factory,
    PageImageCollection,
    page_assembler_step_factory,
    page_text_line_label_step_factory,
    page_distortion_step_factory,
    PageDistortionStepOutput,
    page_resizing_step_factory,
    PageResizingStepOutput,
    page_cropping_step_factory,
    PageCroppingStepOutput,
    pipeline_step_collection_factory,
)

from tests.opt import write_image, write_json


@pytest.mark.local
def test_background():
    pipeline = Pipeline(
        steps=[
            page_shape_step_factory.create(),
            page_background_step_factory.create({
                'image_configs': [{
                    'type': 'combiner',
                    'config': {
                        'image_meta_folder': '$VKIT_DATA/vkit/engine/image/combiner/image_meta'
                    }
                }]
            }),
        ],
        post_processor=bypass_post_processor_factory.create(),
    )
    rng = default_rng(0)
    result = pipeline.run(rng)
    assert result


@pytest.mark.local
def test_page_layout():
    from vkit.pipeline.text_detection.page_layout import page_layout_step_factory, PageLayoutStepOutput, Sequence, Box

    pipeline = Pipeline(
        steps=[
            page_shape_step_factory.create(),
            page_layout_step_factory.create(),
        ],
        post_processor=bypass_post_processor_factory.create(),
    )
    for seed in range(10):
        rng = default_rng(seed)
        result = pipeline.run(rng)
        page_layout_output: PageLayoutStepOutput = result.key_to_value['page_layout_step']
        page_layout = page_layout_output.page_layout
        boxes = list(page_layout_output.debug_normal_grids)
        color = ['green'] * len(boxes)
        if page_layout_output.debug_large_text_line_gird:
            boxes.append(page_layout_output.debug_large_text_line_gird)
            color.append('red')

        image = Image.from_shape((page_layout.height, page_layout.width))
        painter = Painter.create(image)
        painter.paint_boxes(boxes, color=color)
        painter.paint_boxes(
            [layout_text_line.box for layout_text_line in page_layout.layout_text_lines],
            color='white',
            border_thickness=2,
        )
        painter.paint_boxes(
            [layout_image.box for layout_image in page_layout.layout_images],
            color='red',
            border_thickness=2,
        )
        write_image(f'grids_{seed}.jpg', painter.image)


# @pytest.mark.local
# def test_page_text_line():
#     pipeline = Pipeline(
#         steps=[
#             page_shape_step_factory.create(),
#             page_layout_step_factory.create(),
#             page_text_line_step_factory.create({
#                 'lexicon_collection_json':
#                     '$VKIT_PRIVATE_DATA/vkit_lexicon/lexicon_collection_combined/chinese.json',
#                 'font_collection_folder':
#                     '$VKIT_PRIVATE_DATA/vkit_font/font_collection',
#                 'char_sampler_configs': [{
#                     "weight": 1,
#                     "type": "corpus",
#                     "config": {
#                         "txt_files": ["$VKIT_PRIVATE_DATA/char_sampler/corp-address-debug.txt"]
#                     }
#                 }, {
#                     "weight": 1,
#                     "type": "corpus",
#                     "config": {
#                         "txt_files": ["$VKIT_PRIVATE_DATA/char_sampler/corp-name-debug.txt"]
#                     }
#                 }],
#                 'font_configs': [{
#                     "type": "freetype_default",
#                     "weight": 1
#                 }, {
#                     "type": "freetype_lcd",
#                     "weight": 1
#                 }, {
#                     "type": "freetype_monochrome",
#                     "weight": 1
#                 }],
#             })
#         ],
#         post_processor=bypass_post_processor_factory.create(),
#     )

#     for seed in range(3):
#         rng = default_rng(seed)
#         result = pipeline.run(rng)
#         page_text_line_collection: PageTextLineCollection = result.key_to_value[
#             'page_text_line_step'].page_text_line_collection

#         image = Image.from_shape(
#             (page_text_line_collection.height, page_text_line_collection.width),
#             num_channels=3,
#         )
#         for text_line in page_text_line_collection.text_lines:
#             text_line.box.fill_image(image, text_line.image)
#         write_image(f'text_line_{seed}.jpg', image)

#         image = Image.from_shape(
#             (page_text_line_collection.height, page_text_line_collection.width),
#             num_channels=3,
#         )
#         for text_line in page_text_line_collection.text_lines:
#             for char_box in text_line.char_boxes:
#                 char_box.box.fill_image(image, (255, 0, 0))
#         write_image(f'char_mask_{seed}.jpg', image)


@pytest.mark.local
def test_page():

    pipeline = Pipeline(
        steps=pipeline_step_collection_factory.create([
            {
                'name': 'text_detection.page_shape_step',
            },
            {
                'name': 'text_detection.page_background_step',
                'config': {
                    'image_configs': [{
                        'type': 'combiner',
                        'config': {
                            'image_meta_folder': '$VKIT_DATA/vkit/engine/image/combiner/image_meta'
                        }
                    }]
                },
            },
            {
                'name': 'text_detection.page_layout_step',
            },
            {
                'name': 'text_detection.page_image_step',
                'config': {
                    'image_configs': [{
                        'type': 'selector',
                        'config': {
                            'image_folders': [
                                '$VKIT_PRIVATE_DATA/dataset_synthetext/bg_data/no_background_text_images'  # noqa
                            ],
                        }
                    }]
                },
            },
            {
                'name': 'text_detection.page_qrcode_step',
            },
            {
                'name': 'text_detection.page_barcode_step',
            },
            {
                'name': 'text_detection.page_seal_impresssion_step',
                'config': {
                    'seal_impression_configs': [{
                        'type': 'ellipse',
                        'config': {
                            'icon_image_folders': [
                                '$VKIT_ARTIFACT_PACK/non_text_symbol_image/material-design-icon',
                                '$VKIT_ARTIFACT_PACK/non_text_symbol_image/noto-regular-emoji',
                            ],
                        }
                    }]
                },
            },
            {
                'name': 'text_detection.page_text_line_step',
                'config': {
                    'lexicon_collection_json':
                        '$VKIT_PRIVATE_DATA/vkit_lexicon/lexicon_collection_combined/chinese.json',
                    'font_collection_folder':
                        '$VKIT_PRIVATE_DATA/vkit_font/font_collection',
                    'char_sampler_configs': [{
                        "weight": 1,
                        "type": "corpus",
                        "config": {
                            "txt_files": ["$VKIT_PRIVATE_DATA/char_sampler/debug.txt"]
                        }
                    }],
                    'font_configs': [{
                        "type": "freetype_default",
                        "weight": 1
                    }, {
                        "type": "freetype_monochrome",
                        "weight": 1
                    }],
                    'return_font_variant':
                        False,
                },
            },
            {
                'name': 'text_detection.page_non_text_symbol_step',
                'config': {
                    'symbol_image_folders': ['$VKIT_ARTIFACT_PACK/non_text_symbol_image']
                },
            },
            {
                'name': 'text_detection.page_text_line_bounding_box_step',
            },
            {
                'name': 'text_detection.page_text_line_label_step',
            },
            {
                'name': 'text_detection.page_assembler_step',
            },
            {
                'name': 'text_detection.page_distortion_step',
                'config': {
                    'random_distortion_factory_config': {
                        'disabled_policy_names': [
                            'defocus_blur',
                            'zoom_in_blur',
                        ],
                        'force_post_uniform_rotate': True,
                    },
                    'enable_distorted_char_heights_debug': True,
                    'enable_distorted_text_line_heights_debug': True,
                    # 'debug_random_distortion': True,
                },
            },
            {
                'name': 'text_detection.page_resizing_step',
            },
            {
                'name': 'text_detection.page_cropping_step',
                'config': {
                    'core_size': 400,
                    'pad_size': 100,
                },
            },
        ]),
        post_processor=bypass_post_processor_factory.create(),
    )

    # for seed in (3,):
    for seed in range(10):
        print(seed)
        rng = default_rng(seed)
        result = pipeline.run(rng)

        output: PageDistortionStepOutput = result.key_to_value['page_distortion_step']
        image = output.page_image
        write_image(f'page_{seed}.jpg', image)

        vis_image = image.copy()

        page_distorted_char_mask = output.page_char_mask
        assert page_distorted_char_mask
        page_distorted_char_mask.fill_image(vis_image, (255, 0, 0), 0.5)
        write_image(f'page_{seed}_char_mask.jpg', vis_image)

        painter = Painter.create(image)
        page_distorted_char_height_score_map = output.page_char_height_score_map
        assert page_distorted_char_height_score_map
        painter.paint_score_map(page_distorted_char_height_score_map)
        write_image(f'page_{seed}_char_score_map.jpg', painter.image)

        debug_image = output.page_char_heights_debug_image
        assert debug_image
        write_image(f'page_{seed}_char_score_map_debug.jpg', debug_image)

        page_distorted_text_line_mask = output.page_text_line_mask
        assert page_distorted_text_line_mask
        page_distorted_text_line_mask.fill_image(vis_image, (255, 0, 0), 0.5)
        write_image(f'page_{seed}_text_line_mask.jpg', vis_image)

        painter = Painter.create(image)
        page_distorted_text_line_height_score_map = output.page_text_line_height_score_map
        assert page_distorted_text_line_height_score_map
        painter.paint_score_map(page_distorted_text_line_height_score_map)
        write_image(f'page_{seed}_text_line_score_map.jpg', painter.image)

        debug_image = output.page_text_line_heights_debug_image
        assert debug_image
        write_image(f'page_{seed}_text_line_score_map_debug.jpg', debug_image)

        from typing import Dict, Any
        from vkit.engine.distortion.geometric.grid_rendering.interface import DistortionStateImageGridBased
        from vkit.engine.distortion.geometric.grid_rendering.visualization import visualize_image_grid
        from vkit.element import Box
        # debug = output.page_random_distortion_debug
        # assert debug
        # distortion_images = debug.distortion_images
        # distortion_state = None
        # if debug.distortion_states:
        #     distortion_state = debug.distortion_states[-1]
        # if isinstance(distortion_state, DistortionStateImageGridBased):
        #     src_grid_image = visualize_image_grid(distortion_state.src_image_grid)
        #     if len(distortion_images) > 1:
        #         vis_image = distortion_images[-2]
        #     else:
        #         vis_image = debug['image']
        #     Box.from_shapable(vis_image).fill_image(vis_image, src_grid_image, alpha=0.5)
        #     write_image(f'page_{seed}_src_grid_image_debug.jpg', vis_image)

        #     vis_image = image.copy()
        #     dst_grid_image = visualize_image_grid(distortion_state.dst_image_grid)
        #     Box.from_shapable(vis_image).fill_image(vis_image, dst_grid_image, alpha=0.5)
        #     write_image(f'page_{seed}_dst_grid_image_debug.jpg', vis_image)

        output2: PageResizingStepOutput = result.key_to_value['page_resizing_step']
        image = output2.page_image
        write_image(f'page_resized_{seed}.jpg', image)

        vis_image = image.copy()

        page_distorted_char_mask = output2.page_char_mask
        assert page_distorted_char_mask
        page_distorted_char_mask.fill_image(vis_image, (255, 0, 0), 0.5)
        write_image(f'page_resized_{seed}_char_mask.jpg', vis_image)

        painter = Painter.create(image)
        page_distorted_char_height_score_map = output2.page_char_height_score_map
        assert page_distorted_char_height_score_map
        painter.paint_score_map(page_distorted_char_height_score_map)
        write_image(f'page_resized_{seed}_char_score_map.jpg', painter.image)

        page_distorted_text_line_mask = output2.page_text_line_mask
        assert page_distorted_text_line_mask
        page_distorted_text_line_mask.fill_image(vis_image, (255, 0, 0), 0.5)
        write_image(f'page_resized_{seed}_text_line_mask.jpg', vis_image)

        painter = Painter.create(image)
        page_distorted_text_line_height_score_map = output2.page_text_line_height_score_map
        assert page_distorted_text_line_height_score_map
        painter.paint_score_map(page_distorted_text_line_height_score_map)
        write_image(f'page_resized_{seed}_text_line_score_map.jpg', painter.image)

        # output3: PageCroppingStepOutput = result.key_to_value['page_cropping_step']
        # for idx, cropped_page in enumerate(output3.cropped_pages):
        #     write_image(f'page_cropped_{seed}_{idx}_image.jpg', cropped_page.page_image)
        #     painter = Painter.create(cropped_page.page_image)
        #     painter.paint_mask(cropped_page.page_text_line_mask)
        #     write_image(f'page_cropped_{seed}_{idx}_mask.jpg', painter.image)
        #     painter = Painter.create(cropped_page.page_image)
        #     painter.paint_score_map(cropped_page.page_text_line_height_score_map)
        #     write_image(f'page_cropped_{seed}_{idx}_score_map.jpg', painter.image)

        #     downsampled_label = cropped_page.downsampled_label
        #     assert downsampled_label

        #     page_downsampled_text_line_mask = downsampled_label.page_text_line_mask
        #     assert page_downsampled_text_line_mask
        #     page_downsampled_text_line_height_score_map = downsampled_label.page_text_line_height_score_map
        #     assert page_downsampled_text_line_height_score_map

        #     painter = Painter.create(page_downsampled_text_line_mask)
        #     painter.paint_mask(page_downsampled_text_line_mask)
        #     write_image(f'page_cropped_{seed}_{idx}_ds_mask.jpg', painter.image)

        #     painter = Painter.create(page_downsampled_text_line_height_score_map)
        #     painter.paint_score_map(page_downsampled_text_line_height_score_map)
        #     write_image(f'page_cropped_{seed}_{idx}_ds_score_map.jpg', painter.image)

        # page: Page = result.key_to_value['page']
        # write_image(f'page_{seed}.jpg', page.image)

        # red = (255, 0, 0)
        # alpha = 0.5

        # image = page.image.copy()
        # page_image_collection = page.page_image_collection
        # for page_image in page_image_collection.page_images:
        #     page_image.box.fill_image(image, red, alpha)
        # write_image(f'page_image_{seed}.jpg', image)

        # image = page.image.copy()
        # page_text_line_collection = page.page_text_line_collection
        # for text_line in page_text_line_collection.text_lines:
        #     text_line.box.fill_image(image, red, alpha)
        # write_image(f'page_text_line_{seed}.jpg', image)

        # image = page.image.copy()
        # for text_line in page_text_line_collection.text_lines:
        #     for char_box in text_line.char_boxes:
        #         char_box.box.fill_image(image, red, alpha)
        # write_image(f'page_char_{seed}.jpg', image)

        # image = page.image.copy()
        # page_text_line_mask: Mask = result.key_to_value['page_text_line_mask']
        # page_text_line_mask.fill_image(image, (255, 0, 0), 0.5)
        # write_image(f'page_text_line_mask_{seed}.jpg', image)

        # image = page.image.copy()
        # page_text_line_and_boundary_mask: Mask = result.key_to_value[
        #     'page_text_line_and_boundary_mask']
        # page_text_line_and_boundary_mask.fill_image(image, (255, 0, 0), 0.5)
        # write_image(f'page_text_line_and_boundary_mask_{seed}.jpg', image)

        # image = page.image.copy()
        # page_text_line_boundary_mask: Mask = result.key_to_value['page_text_line_boundary_mask']
        # page_text_line_boundary_mask.fill_image(image, (255, 0, 0), 0.5)
        # write_image(f'page_text_line_boundary_mask_{seed}.jpg', image)

        # page_text_line_boundary_score_map: ScoreMap = \
        #     result.key_to_value['page_text_line_boundary_score_map']
        # image = Image.from_shapable(page_text_line_boundary_score_map)
        # image.mat = (page_text_line_boundary_score_map.mat * 255).astype(np.uint8)
        # write_image(f'page_text_line_boundary_score_map_{seed}.jpg', image)

        # image = page.image.copy()
        # page_text_line_boundary_score_map.fill_image(image, (255, 0, 0))
        # write_image(f'page_text_line_boundary_score_map2_{seed}.jpg', image)

        # if page_text_line_step.config.return_font_variant:
        #     image = page.image.copy()
        #     write_image(
        #         f'debug_text_lines/{seed}/image.png',
        #         image,
        #     )

        #     boxes = [text_line.box for text_line in page_text_line_collection.text_lines]
        #     painter = Painter.create(image)
        #     painter.paint_boxes(boxes, enable_index=True)
        #     write_image(
        #         f'debug_text_lines/{seed}/index.png',
        #         painter.image,
        #     )

        #     for idx, text_line in enumerate(page_text_line_collection.text_lines):
        #         write_image(
        #             f'debug_text_lines/{seed}/{idx}.png',
        #             text_line.image,
        #         )
        #         painter = Painter.create(text_line.mask)
        #         painter.paint_mask(text_line.mask, alpha=1.0)
        #         write_image(
        #             f'debug_text_lines/{seed}/{idx}_mask.png',
        #             painter.image,
        #         )
        #         if text_line.score_map:
        #             image = text_line.image.copy()
        #             text_line.score_map.fill_image(image, (255, 0, 0))
        #             write_image(
        #                 f'debug_text_lines/{seed}/{idx}_score_map.png',
        #                 image,
        #             )

        #         assert text_line.font_variant
        #         font_variant_data = cattrs.unstructure(text_line.font_variant)
        #         font_variant_data['font_file'] = str(font_variant_data['font_file'])
        #         write_json(
        #             f'debug_text_lines/{seed}/{idx}_font_variant.json',
        #             font_variant_data,
        #         )


def debug_adaptive_scaling_pipeline():
    steps_json = '$VKIT_ARTIFACT_PACK/pipeline/text_detection/dev_adaptive_scaling_dataset_steps.json'
    pipeline = Pipeline(
        steps=pipeline_step_collection_factory.create(steps_json),
        post_processor=bypass_post_processor_factory.create(),
    )

    # Profiling.
    # from pyinstrument import Profiler
    # p = Profiler(async_mode='disabled')
    # p.start()
    # for rng_seed in range(0, 1000):
    #     print('rng_seed =', rng_seed)
    #     pipeline.run(default_rng(rng_seed))
    # p.stop()
    # p.print()
    # breakpoint()

    # The one with labeling issue.
    rng = default_rng()
    rng.bit_generator.state = {
        "bit_generator": "PCG64",
        "state": {
            "state": 265664896678725395205730166041826474231,
            "inc": 264764775878313123560428660725838731763
        },
        "has_uint32": 0,
        "uinteger": 1688944281
    }
    state = pipeline.run(rng=rng)

    output: PageDistortionStepOutput = state.key_to_value['page_distortion_step']
    image = output.page_image
    write_image(f'page.png', image)

    page_distorted_char_mask = output.page_char_mask
    assert page_distorted_char_mask
    vis_image = image.copy()
    page_distorted_char_mask.fill_image(vis_image, (255, 0, 0), 0.5)
    write_image(f'page_char_mask.png', vis_image)

    painter = Painter.create(image)
    page_distorted_char_height_score_map = output.page_char_height_score_map
    assert page_distorted_char_height_score_map
    painter.paint_score_map(page_distorted_char_height_score_map)
    write_image(f'page_char_score_map.png', painter.image)

    output2: PageResizingStepOutput = state.key_to_value['page_resizing_step']
    image = output2.page_image
    write_image(f'page_resized.png', image)

    page_distorted_char_mask = output2.page_char_mask
    assert page_distorted_char_mask
    vis_image = image.copy()
    page_distorted_char_mask.fill_image(vis_image, (255, 0, 0), 0.5)
    write_image(f'page_resized_char_mask.png', vis_image)

    painter = Painter.create(image)
    page_distorted_char_height_score_map = output2.page_char_height_score_map
    assert page_distorted_char_height_score_map
    painter.paint_score_map(page_distorted_char_height_score_map)
    write_image(f'page_resized_char_score_map.png', painter.image)

    output3: PageCroppingStepOutput = state.key_to_value['page_cropping_step']
    for idx, cropped_page in enumerate(output3.cropped_pages):
        write_image(f'page_cropped_{idx}_image.png', cropped_page.page_image)
        painter = Painter.create(cropped_page.page_image)
        painter.paint_mask(cropped_page.page_char_mask)
        write_image(f'page_cropped_{idx}_mask.png', painter.image)
        painter = Painter.create(cropped_page.page_image)
        painter.paint_score_map(cropped_page.page_char_height_score_map)
        write_image(f'page_cropped_{idx}_score_map.png', painter.image)

        assert cropped_page.page_char_mask.box
        char_mask_area = cropped_page.page_char_mask.box.area
        char_mask_ratio = (cropped_page.page_char_mask.mat > 0).sum() / char_mask_area
        print(f'{idx}: char_mask_ratio = {char_mask_ratio}')

        downsampled_label = cropped_page.downsampled_label
        assert downsampled_label

        page_downsampled_char_mask = downsampled_label.page_char_mask
        assert page_downsampled_char_mask
        page_downsampled_char_height_score_map = downsampled_label.page_char_height_score_map
        assert page_downsampled_char_height_score_map

        painter = Painter.create(page_downsampled_char_mask)
        painter.paint_mask(page_downsampled_char_mask)
        write_image(f'page_cropped_{idx}_ds_mask.png', painter.image)

        painter = Painter.create(page_downsampled_char_height_score_map)
        painter.paint_score_map(page_downsampled_char_height_score_map)
        write_image(f'page_cropped_{idx}_ds_score_map.png', painter.image)


# @pytest.mark.local
# def debug_font_face_diff_index():
#     import iolite as io

#     font_file = io.file(
#         '$VKIT_ARTIFACT_PACK/font_collection/font/WenQuanYiZenHei.ttc',
#         expandvars=True,
#         exists=True,
#     )

#     import freetype

#     face0 = freetype.Face(str(font_file), index=0)

#     face2 = freetype.Face(str(font_file), index=2)

#     codepoints = []
#     codepoint, index = face2.get_first_char()
#     assert index
#     codepoints.append(codepoint)
#     while index:
#         codepoint, index = face2.get_next_char(codepoint, index)
#         codepoints.append(codepoint)

#     chars = [chr(cp) for cp in codepoints]
#     breakpoint()

#     base_res = 72
#     font_size = 15

#     face0.set_char_size(
#         width=font_size << 6,
#         height=0,
#         hres=base_res,
#         vres=base_res,
#     )
#     face2.set_char_size(
#         width=font_size << 6,
#         height=0,
#         hres=base_res,
#         vres=base_res,
#     )

#     load_char_flags = freetype.FT_LOAD_RENDER  # type: ignore

#     char = '抑'

#     face0.load_char(char, load_char_flags)
#     glyph = face0.glyph
#     bitmap = glyph.bitmap

#     height = bitmap.rows
#     width = bitmap.width
#     # assert width == bitmap.pitch
#     print('face0, height =', height, 'width =', width, 'pitch =', bitmap.pitch)

#     face2.load_char(char, load_char_flags)
#     glyph = face2.glyph
#     bitmap = glyph.bitmap

#     height = bitmap.rows
#     width = bitmap.width
#     # assert width == bitmap.pitch
#     print('face2, height =', height, 'width =', width, 'pitch =', bitmap.pitch)
