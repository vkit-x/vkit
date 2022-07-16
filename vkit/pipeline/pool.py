from typing import (
    Generic,
    TypeVar,
    Mapping,
    Any,
    Sequence,
    Optional,
    List,
)
from queue import Empty
from multiprocessing import JoinableQueue, Process
import logging

from numpy.random import (
    SeedSequence,
    default_rng,
)

from .interface import Pipeline

logger = logging.getLogger(__name__)

_T_OUTPUT = TypeVar('_T_OUTPUT')


class PipelinePoolRunner(Generic[_T_OUTPUT]):

    def __init__(
        self,
        process_idx: int,
        pipeline: Pipeline[_T_OUTPUT],
        rng_state: Mapping[str, Any],
        num_runs_reset_rng: Optional[int],
    ):
        self.process_idx = process_idx
        self.pipeline = pipeline

        self.rng_state = rng_state
        self.rng = default_rng()
        self.reset_rng()

        self.num_runs_reset_rng = num_runs_reset_rng
        self.run_idx = 0

    def reset_rng(self):
        self.rng.bit_generator.state = self.rng_state
        logger.debug(
            f'Reset pipeline process_idx={self.process_idx} '
            f'rng_state to {self.rng.bit_generator.state}'
        )

    def run(self):
        output: Optional[_T_OUTPUT] = None

        while True:
            cur_rng_state = self.rng.bit_generator.state
            try:
                output = self.pipeline.run(self.rng)
                break
            except Exception:
                logger.exception(
                    f'pipeline.run process_idx={self.process_idx} failed with '
                    f'rng_state={cur_rng_state}, retrying...'
                )
                if self.rng.bit_generator.state == cur_rng_state:
                    # Force to change rng state.
                    self.rng.random()

        assert output is not None

        self.run_idx += 1
        if self.num_runs_reset_rng and self.run_idx % self.num_runs_reset_rng == 0:
            logger.debug(f'pipeline.run run_idx={self.run_idx}, hit reset_rng')
            self.reset_rng()

        return output


def pipeline_pool_process_target(
    process_idx: int,
    pipeline: Pipeline[_T_OUTPUT],
    rng_state: Mapping[str, Any],
    num_runs_per_worker: int,
    num_runs_reset_rng: Optional[int],
    joinable_queues: Sequence[JoinableQueue[_T_OUTPUT]],
):
    pipeline_pool_runner = PipelinePoolRunner(
        process_idx=process_idx,
        pipeline=pipeline,
        rng_state=rng_state,
        num_runs_reset_rng=num_runs_reset_rng,
    )

    while True:
        for joinable_queue_idx, joinable_queue in enumerate(joinable_queues):
            logger.debug(
                f'process_idx={process_idx} waiting for '
                f'joinable_queue={joinable_queue_idx} ...'
            )
            joinable_queue.join()

            logger.debug(
                f'process_idx={process_idx} start filling '
                f'joinable_queue={joinable_queue_idx} ...'
            )
            for run_idx in range(num_runs_per_worker):
                joinable_queue.put(pipeline_pool_runner.run())
                logger.debug(
                    f'process_idx={process_idx} filled run_idx={run_idx} to '
                    f'joinable_queue={joinable_queue_idx}'
                )


class PipelinePool(Generic[_T_OUTPUT]):

    def __init__(
        self,
        pipeline: Pipeline[_T_OUTPUT],
        rng_seed: int,
        num_workers: int,
        num_runs_per_worker: int,
        num_runs_reset_rng: Optional[int] = None,
        num_joinable_queues: int = 2,
        get_timeout: int = 20,
    ):
        assert num_workers > 0
        assert num_runs_per_worker > 0
        if num_runs_reset_rng is not None:
            assert num_runs_reset_rng > 0

        self.pipeline = pipeline
        self.rng_seed = rng_seed
        self.num_workers = num_workers
        self.num_runs_per_worker = num_runs_per_worker
        self.num_runs_reset_rng = num_runs_reset_rng
        self.num_joinable_queues = num_joinable_queues

        self.joinable_queues: Sequence[JoinableQueue[_T_OUTPUT]] = []
        self.processes: Sequence[Process] = []
        self.reset()

        self.num_runs_per_joinable_queue = num_workers * num_runs_per_worker
        self.get_timeout = get_timeout
        self.joinable_queue_idx = 0
        self.run_idx = 0

    def reset(self):
        # Killing all processes.
        for process_idx, process in enumerate(self.processes):
            logger.debug(f'Killing process_idx={process_idx} ...')
            process.kill()
            process.close()
        # Then closing all joinable queues.
        for joinable_queue in self.joinable_queues:
            joinable_queue.close()

        self.joinable_queues = [JoinableQueue() for _ in range(self.num_joinable_queues)]
        logger.debug(f'{self.num_joinable_queues} joinable queues initialized.')

        processes: List[Process] = []
        seed_sequences = SeedSequence(self.rng_seed).spawn(self.num_workers)
        for process_idx, seed_sequence in enumerate(seed_sequences):
            logger.debug(f'Creating process_idx={process_idx} ...')
            process = Process(
                target=pipeline_pool_process_target,
                name=f'pipeline_pool_process_{process_idx}',
                kwargs={
                    'process_idx': process_idx,
                    'pipeline': self.pipeline,
                    'rng_state': default_rng(seed_sequence).bit_generator.state,
                    'num_runs_per_worker': self.num_runs_per_worker,
                    'num_runs_reset_rng': self.num_runs_reset_rng,
                    'joinable_queues': self.joinable_queues,
                },
                daemon=True,
            )
            process.start()
            processes.append(process)
        self.processes = processes
        logger.debug('All resources reset.')

    def run(self):
        output: Optional[_T_OUTPUT] = None

        joinable_queue = self.joinable_queues[self.joinable_queue_idx]
        try:
            output = joinable_queue.get(timeout=self.get_timeout)
            joinable_queue.task_done()
        except Empty:
            logger.error(
                'taking too long to get output from '
                f'joinable_queue={self.joinable_queue_idx}, something is wrong.'
            )
            raise
        assert output is not None

        self.run_idx += 1
        if self.run_idx >= self.num_runs_per_joinable_queue:
            self.joinable_queue_idx = (self.joinable_queue_idx + 1) % self.num_joinable_queues
            self.run_idx = 0
            logger.debug(f'Switch to joinable_queue={self.joinable_queue_idx}')

        return output
