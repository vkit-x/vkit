import time
from multiprocessing import JoinableQueue, Process
import numpy as np

from vkit.pipeline import (
    Pipeline,
    PipelinePool,
    bypass_post_processor_factory,
    pipeline_step_collection_factory,
)


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
        num_workers=2,
        num_runs_per_worker=2,
    )

    shapes0 = []
    for _ in range(4):
        state = pipeline_pool.run()
        page_shape_step = state.key_to_value['page_shape_step']
        shapes0.append((page_shape_step.height, page_shape_step.width))

    pipeline_pool.reset()
    print('!!! reset !!!')

    shapes1 = []
    for _ in range(4):
        state = pipeline_pool.run()
        page_shape_step = state.key_to_value['page_shape_step']
        shapes1.append((page_shape_step.height, page_shape_step.width))

    assert set(shapes0) == set(shapes1)

    pipeline_pool.cleanup()

    print('!!! set num_runs_reset_rng !!!')
    pipeline_pool = PipelinePool(
        pipeline=pipeline,
        rng_seed=1234,
        num_workers=2,
        num_runs_per_worker=4,
        num_runs_reset_rng=2,
    )
    shapes = []
    for _ in range(8):
        state = pipeline_pool.run()
        page_shape_step = state.key_to_value['page_shape_step']
        shapes.append((page_shape_step.height, page_shape_step.width))
    assert len(set(shapes)) == 4
