from typing import (
    Generic,
    TypeVar,
    Mapping,
    Any,
    Sequence,
    Optional,
    List,
    Dict,
)
from queue import Empty
from multiprocessing import (
    Manager,
    Queue,
    Process,
    Lock,
    log_to_stderr,
)
import os
import time
import logging

from numpy.random import (
    SeedSequence,
    default_rng,
)

from .interface import Pipeline

_T_OUTPUT = TypeVar('_T_OUTPUT')


class PipelinePoolRunner(Generic[_T_OUTPUT]):

    def __init__(
        self,
        process_idx: int,
        logger: logging.Logger,
        pipeline: Pipeline[_T_OUTPUT],
        rng_state: Mapping[str, Any],
        num_runs_reset_rng: Optional[int],
    ):
        self.process_idx = process_idx
        self.logger = logger

        self.pipeline = pipeline

        self.rng_state = rng_state
        self.rng = default_rng()
        self.reset_rng()

        self.num_runs_reset_rng = num_runs_reset_rng
        self.rng_run_idx = 0

    def reset_rng(self):
        self.rng.bit_generator.state = self.rng_state
        self.rng_run_idx = 0
        self.logger.debug(
            f'Reset pipeline process_idx={self.process_idx} '
            f'rng_state to {self.rng.bit_generator.state} '
            'and run_idx to 0'
        )

    def run(self):
        output: Optional[_T_OUTPUT] = None

        while True:
            cur_rng_state = self.rng.bit_generator.state
            try:
                output = self.pipeline.run(self.rng)
                self.logger.debug(
                    f'pipeline.run process_idx={self.process_idx} with '
                    f'rng_state={cur_rng_state} generates output={output}'
                )
                break
            except Exception:
                self.logger.exception(
                    f'pipeline.run process_idx={self.process_idx} failed with '
                    f'rng_state={cur_rng_state}, retrying...'
                )
                if self.rng.bit_generator.state == cur_rng_state:
                    # Force to change rng state.
                    self.rng.random()

        assert output is not None

        self.rng_run_idx += 1
        if self.num_runs_reset_rng and self.rng_run_idx % self.num_runs_reset_rng == 0:
            self.logger.debug(f'pipeline.run rng_run_idx={self.rng_run_idx}, hit reset_rng')
            self.reset_rng()

        return output


def pipeline_pool_process_target(
    process_idx: int,
    pipeline: Pipeline[_T_OUTPUT],
    rng_state: Mapping[str, Any],
    num_runs_per_process: int,
    num_runs_reset_rng: Optional[int],
    queues: Sequence['Queue[_T_OUTPUT]'],
    process_status_for_queues: List[Dict[int, bool]],
    process_status_for_queues_lock: Any,
):
    logger = log_to_stderr(os.getenv('LOGGING_LEVEL'))

    pipeline_pool_runner = PipelinePoolRunner(
        process_idx=process_idx,
        logger=logger,
        pipeline=pipeline,
        rng_state=rng_state,
        num_runs_reset_rng=num_runs_reset_rng,
    )

    while True:
        for queue_idx, queue in enumerate(queues):
            logger.debug(f'process_idx={process_idx} waiting for queue={queue_idx} ...')
            while True:
                process_status_for_queues_lock.acquire()
                process_status = process_status_for_queues[queue_idx]
                process_status_for_queues_lock.release()

                if process_status[process_idx]:
                    break

                logger.debug(f'queue={queue_idx} is NOT ready for process, waiting ...')
                time.sleep(0.05)

            logger.debug(f'process_idx={process_idx} start filling queue={queue_idx} ...')
            for process_run_idx in range(num_runs_per_process):
                output = pipeline_pool_runner.run()
                queue.put(output)
                logger.debug(
                    f'process_idx={process_idx} filled process_run_idx={process_run_idx} '
                    f'output={output} to '
                    f'queue={queue_idx}'
                )

            process_status_for_queues_lock.acquire()
            process_status_for_queues[queue_idx][process_idx] = False
            process_status = process_status_for_queues[queue_idx]
            process_status_for_queues_lock.release()
            logger.debug(
                f'process_idx={process_idx} finished filling '
                f'queue={queue_idx}, '
                f'change process_status to {process_status}'
            )


logger = logging.getLogger(__name__)


class PipelinePool(Generic[_T_OUTPUT]):

    def __init__(
        self,
        pipeline: Pipeline[_T_OUTPUT],
        rng_seed: int,
        num_processes: int,
        num_runs_per_process: int,
        num_runs_reset_rng: Optional[int] = None,
        num_queues: int = 2,
        get_timeout: int = 20,
    ):
        assert num_processes > 0
        assert num_runs_per_process > 0
        if num_runs_reset_rng is not None:
            assert num_runs_reset_rng > 0

        self.pipeline = pipeline
        self.rng_seed = rng_seed
        self.num_processes = num_processes
        self.num_runs_per_process = num_runs_per_process
        self.num_runs_reset_rng = num_runs_reset_rng
        self.num_queues = num_queues

        self.queues: Sequence['Queue[_T_OUTPUT]'] = []
        self.manager = Manager()
        try:
            self.manager.start()
        except Exception:
            # Hack.
            time.sleep(0.1)
        self.process_status_for_queues: List[Dict[int, bool]] = self.manager.list()  # type: ignore
        self.process_status_for_queues_lock = Lock()
        self.processes: Sequence[Process] = []
        self.reset()

        self.num_runs_per_queue = num_processes * num_runs_per_process
        self.get_timeout = get_timeout
        self.queue_idx = 0
        self.run_idx = 0

    def cleanup(self):
        # Killing all processes.
        for process_idx, process in enumerate(self.processes):
            logger.debug(f'Killing process_idx={process_idx} ...')
            try:
                process.kill()
                while process.is_alive():
                    logger.debug(f'Waiting for process_idx={process_idx} to be killed ...')
                    time.sleep(0.05)
                process.close()
            except Exception:
                logger.exception('Cannot kill process.')
        self.processes = []

        # Then closing all queues.
        for queue in self.queues:
            queue.close()
        self.queues = []

        # Then the shared list and manager.
        self.process_status_for_queues = []
        try:
            self.manager.shutdown()
        except Exception:
            logger.warning('Cannot shutdown manager.')

    def reset(self):
        self.cleanup()

        self.queues = [Queue() for _ in range(self.num_queues)]
        self.manager = Manager()
        try:
            self.manager.start()
        except Exception:
            # Hack.
            time.sleep(0.1)
        process_status_for_queues = []
        for _ in range(self.num_queues):
            process_status = dict((process_idx, True) for process_idx in range(self.num_processes))
            process_status_for_queues.append(self.manager.dict(process_status))
        self.process_status_for_queues = self.manager.list(  # type: ignore
            process_status_for_queues
        )
        self.process_status_for_queues_lock = Lock()
        logger.debug(f'{self.num_queues} queues initialized.')

        processes: List[Process] = []
        seed_sequences = SeedSequence(self.rng_seed).spawn(self.num_processes)
        for process_idx, seed_sequence in enumerate(seed_sequences):
            logger.debug(f'Creating process_idx={process_idx} ...')
            process = Process(
                target=pipeline_pool_process_target,
                name=f'pipeline_pool_process_{process_idx}',
                kwargs={
                    'process_idx': process_idx,
                    'pipeline': self.pipeline,
                    'rng_state': default_rng(seed_sequence).bit_generator.state,
                    'num_runs_per_process': self.num_runs_per_process,
                    'num_runs_reset_rng': self.num_runs_reset_rng,
                    'queues': self.queues,
                    'process_status_for_queues': self.process_status_for_queues,
                    'process_status_for_queues_lock': self.process_status_for_queues_lock,
                },
                daemon=True,
            )
            process.start()
            processes.append(process)
        self.processes = processes

        self.queue_idx = 0
        self.run_idx = 0

        logger.debug('All resources reset.')

    def run(self):
        time_begin = time.time()
        queue_is_ready = False
        while time.time() - time_begin < self.get_timeout:
            self.process_status_for_queues_lock.acquire()
            process_status = self.process_status_for_queues[self.queue_idx]
            self.process_status_for_queues_lock.release()

            if not any(process_status.values()):
                queue_is_ready = True
                break

            logger.debug(f'queue={self.queue_idx} is NOT ready for consumer, waiting ...')
            time.sleep(0.05)

        assert queue_is_ready
        logger.debug(f'Can consume queue={self.queue_idx} now!')

        output: Optional[_T_OUTPUT] = None
        queue = self.queues[self.queue_idx]
        try:
            output = queue.get(timeout=self.get_timeout)
        except Empty:
            logger.error(
                'taking too long to get output from '
                f'queue={self.queue_idx}, something is wrong.'
            )
            raise
        assert output is not None

        self.run_idx += 1
        if self.run_idx >= self.num_runs_per_queue:
            logger.debug(f'Make queue={self.queue_idx} ready for processes.')
            self.process_status_for_queues_lock.acquire()
            for process_idx in range(self.num_processes):
                self.process_status_for_queues[self.queue_idx][process_idx] = True
            self.process_status_for_queues_lock.release()

            self.queue_idx = (self.queue_idx + 1) % self.num_queues
            self.run_idx = 0
            logger.debug(f'Switch to queue={self.queue_idx}')

        return output
