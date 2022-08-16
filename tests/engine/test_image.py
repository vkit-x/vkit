# def debug():
#     from vkit.utility import get_data_folder
#     folder = get_data_folder(__file__)

#     from vkit.engine.interface import EngineExecutorFactory

#     image_combiner_engine = EngineExecutorFactory(ImageCombinerEngine).create({
#         'image_meta_folder': f'{folder}/image_meta',
#     })
#     for seed in range(20):
#         image = image_combiner_engine.run(
#             ImageEngineRunConfig(
#                 height=891,
#                 width=630,
#             ),
#             rng=default_rng(seed),
#         )
#         image.to_file(f'{folder}/debug/{seed}.png')
