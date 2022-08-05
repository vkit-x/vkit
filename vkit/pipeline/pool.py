from typing import TypeVar, Generic, Optional
import logging
import time

import attrs
from numpy.random import SeedSequence, default_rng

from vkit.utility import Pool, PoolConfig
from .interface import Pipeline

_T_OUTPUT = TypeVar('_T_OUTPUT')


@attrs.define
class PipelinePoolWorkerConfig(Generic[_T_OUTPUT]):
    pipeline: Pipeline[_T_OUTPUT]
    num_runs_reset_rng: Optional[int]


class PipelinePoolWorker(Generic[_T_OUTPUT]):

    def __init__(
        self,
        process_idx: int,
        seed_sequence: SeedSequence,
        logger: logging.Logger,
        config: PipelinePoolWorkerConfig[_T_OUTPUT],
    ):
        self.process_idx = process_idx
        self.logger = logger

        self.seed_sequence = seed_sequence
        self.rng = default_rng(self.seed_sequence)
        self.rng_run_idx = 0

        self.pipeline = config.pipeline
        self.num_runs_reset_rng = config.num_runs_reset_rng

    def reset_rng(self):
        self.rng = default_rng(self.seed_sequence)
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


class PipelinePool(Generic[_T_OUTPUT]):

    def __init__(
        self,
        pipeline: Pipeline[_T_OUTPUT],
        inventory: int,
        num_processes: int,
        rng_seed: int,
        num_runs_reset_rng: Optional[int] = None,
        timeout: int = 10,
    ):
        self.pool_creator = lambda: Pool(
            config=PoolConfig(
                inventory=inventory,
                num_processes=num_processes,
                pool_worker_class=PipelinePoolWorker[_T_OUTPUT],
                pool_worker_config=PipelinePoolWorkerConfig(
                    pipeline=pipeline,
                    num_runs_reset_rng=num_runs_reset_rng,
                ),
                rng_seed=rng_seed,
                timeout=timeout,
                logging_level=logging.DEBUG,
            )
        )
        self.pool = self.pool_creator()

    def cleanup(self):
        self.pool.cleanup()

    def reset(self):
        self.cleanup()
        time.sleep(2.5)
        self.pool = self.pool_creator()

    def run(self):
        return self.pool.run()
