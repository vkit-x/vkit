import time
from multiprocessing import JoinableQueue, Process
import logging

import numpy as np

from vkit.pipeline import (
    Pipeline,
    PipelinePool,
    bypass_post_processor_factory,
    pipeline_step_collection_factory,
)

logger = logging.getLogger(__name__)


def demo_target(joinable_queue: JoinableQueue, process_idx: int):
    while True:
        print(f'process_idx={process_idx} joining')
        joinable_queue.join()
        for idx in range(10):
            print(f'putting ({process_idx}, {idx})')
            joinable_queue.put((process_idx, idx, np.random.rand(3)))


def debug_joinable_queue():
    joinable_queue = JoinableQueue()

    process0 = Process(
        target=demo_target,
        kwargs={
            'joinable_queue': joinable_queue,
            'process_idx': 0,
        },
        daemon=True,
    )
    process0.start()
    print(process0)

    process0 = Process(
        target=demo_target,
        kwargs={
            'joinable_queue': joinable_queue,
            'process_idx': 1,
        },
        daemon=True,
    )
    process0.start()
    print(process0)

    for _ in range(30):
        process_idx, idx, vec = joinable_queue.get()
        print(f'getting {idx}, {vec} from process_idx={process_idx}')
        joinable_queue.task_done()
        time.sleep(0.5)


def test_pool():
    begin = 1 / 1.4142
    end = 1.4142
    length = end - begin
    num_aspect_ratios = 1000
    aspect_ratios = [
        begin + length / num_aspect_ratios * idx for idx in range(num_aspect_ratios + 1)
    ]
    pipeline = Pipeline(
        steps=pipeline_step_collection_factory.create([
            {
                'name': 'text_detection.page_shape_step',
                'config': {
                    'aspect_ratios': aspect_ratios,
                }
            },
        ]),
        post_processor=bypass_post_processor_factory.create(),
    )

    pipeline_pool = PipelinePool(
        pipeline=pipeline,
        rng_seed=1234,
        num_processes=2,
        num_runs_per_process=4,
    )
    shapes = []
    for _ in range(8):
        state = pipeline_pool.run()
        page_shape_step = state.key_to_value['page_shape_step']
        shapes.append((page_shape_step.height, page_shape_step.width))
    assert len(set(shapes)) == 8

    for _ in range(4):
        state = pipeline_pool.run()
        page_shape_step = state.key_to_value['page_shape_step']
        shapes.append((page_shape_step.height, page_shape_step.width))
    assert len(set(shapes)) > 8

    pipeline_pool.cleanup()

    pipeline_pool = PipelinePool(
        pipeline=pipeline,
        rng_seed=1234,
        num_processes=2,
        num_runs_per_process=2,
        num_runs_reset_rng=1,
    )

    shapes0 = []
    for _ in range(4):
        state = pipeline_pool.run()
        page_shape_step = state.key_to_value['page_shape_step']
        shapes0.append((page_shape_step.height, page_shape_step.width))

    pipeline_pool.reset()

    shapes1 = []
    for _ in range(4):
        state = pipeline_pool.run()
        page_shape_step = state.key_to_value['page_shape_step']
        shapes1.append((page_shape_step.height, page_shape_step.width))

    assert set(shapes0) == set(shapes1)
    assert len(set(shapes0)) == 2

    pipeline_pool.cleanup()

    print('!!! set num_runs_reset_rng !!!')
    pipeline_pool = PipelinePool(
        pipeline=pipeline,
        rng_seed=1234,
        num_processes=2,
        num_runs_per_process=4,
        num_runs_reset_rng=2,
    )
    shapes = []
    for _ in range(8):
        state = pipeline_pool.run()
        page_shape_step = state.key_to_value['page_shape_step']
        shapes.append((page_shape_step.height, page_shape_step.width))
    assert len(set(shapes)) == 4

    pipeline_pool.cleanup()


# class ProcState:
#     ID: int = -1
#     OFFSET: int = -1

# def initializer():
#     from multiprocessing import current_process
#     proc = current_process()
#     ProcState.ID = proc._identity[0]
#     ProcState.OFFSET = 0

# def proc_producer(_):
#     ProcState.OFFSET += 1
#     result = f'{ProcState.ID}-{ProcState.OFFSET}'
#     print(f'{result} generated.')
#     return result

# class TickState:
#     GO: bool = True

# def generate_ticks():
#     import threading
#     import os
#     idx = 0
#     while True:
#         while not TickState.GO:
#             print('no go', threading.get_ident())
#             time.sleep(0.5)
#         print('go', threading.get_ident())
#         yield idx
#         idx += 1
#         time.sleep(0.1)

# class Ticker:
#     pass

# def debug_mp():
#     import os
#     import threading
#     from multiprocessing import Pool

#     print('main', os.getpid())
#     with Pool(processes=2, initializer=initializer) as pool:
#         it = pool.imap_unordered(proc_producer, generate_ticks())
#         print('ok')
#         while True:
#             print('loop', threading.get_ident(), next(it))
#             print('loop', threading.get_ident(), it._items)
#             if len(it._items) > 30:
#                 TickState.GO = False
#             else:
#                 TickState.GO = True
#             time.sleep(10.0)
