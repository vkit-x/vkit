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
from typing import Optional, Protocol, TypeVar, Generic, Type, Any
import multiprocessing
import multiprocessing.util
import multiprocessing.process
import threading
import logging
import time
import atexit

import attrs
from numpy.random import SeedSequence
import psutil

logger = logging.getLogger(__name__)

_T_CONFIG = TypeVar('_T_CONFIG', contravariant=True)
_T_OUTPUT = TypeVar('_T_OUTPUT', covariant=True)


class PoolWorkerProtocol(Protocol[_T_CONFIG, _T_OUTPUT]):

    def __init__(
        self,
        process_idx: int,
        seed_sequence: SeedSequence,
        logger: logging.Logger,
        config: _T_CONFIG,
    ) -> None:
        ...

    def run(self) -> _T_OUTPUT:
        ...


@attrs.define
class PoolConfig(Generic[_T_CONFIG, _T_OUTPUT]):
    inventory: int
    num_processes: int
    pool_worker_class: Type[PoolWorkerProtocol[_T_CONFIG, _T_OUTPUT]]
    pool_worker_config: _T_CONFIG
    schedule_size_min_factor: float = 1.0
    rng_seed: int = 133700
    logging_level: int = logging.INFO
    logging_format: str = '[%(levelname)s/PROCESS_IDX] %(message)s'
    logging_to_stderr: bool = False
    timeout: Optional[int] = None


class PoolWorkerState:
    pool_worker: Optional[PoolWorkerProtocol] = None
    logger: logging.Logger


def pool_worker_initializer(pool_config: PoolConfig, process_counter: Any):
    with process_counter.get_lock():
        process_idx: int = process_counter.value  # type: ignore
        process_counter.value += 1  # type: ignore

    # Overriding logger.
    logger = multiprocessing.get_logger()
    logger_stream_handler = logging.StreamHandler()
    logger_formatter = logging.Formatter(
        pool_config.logging_format.replace('PROCESS_IDX', str(process_idx))
    )
    logger_stream_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_stream_handler)
    logger.setLevel(pool_config.logging_level)

    if pool_config.logging_to_stderr:
        multiprocessing.util._log_to_stderr = True  # type: ignore

    logger.debug(f'process_idx={process_idx}, num_processes={pool_config.num_processes}')

    # Generate seed_sequence.
    seed_sequences = SeedSequence(pool_config.rng_seed)
    seed_sequence = seed_sequences.spawn(pool_config.num_processes)[process_idx]

    # Initialize pool worker.
    logger.debug(f'Initializing process_idx={process_idx} with seed_sequence={seed_sequence}')
    pool_worker = pool_config.pool_worker_class(
        process_idx=process_idx,
        seed_sequence=seed_sequence,
        config=pool_config.pool_worker_config,
        logger=logger,
    )
    PoolWorkerState.pool_worker = pool_worker
    PoolWorkerState.logger = logger
    logger.debug('Initialized.')


def pool_worker_runner(_):
    logger = PoolWorkerState.logger
    logger.debug('Triggered.')

    pool_worker = PoolWorkerState.pool_worker
    assert pool_worker is not None
    result = pool_worker.run()
    logger.debug('Result generated.')

    return result


@attrs.define
class PoolInventoryState:
    cond: threading.Condition
    inventory: int
    num_scheduled: int
    inventory_target: int
    abort: bool

    def predicate(self):
        return self.abort or (self.inventory + self.num_scheduled < self.inventory_target)

    def __repr__(self):
        return (
            'PoolInventoryState('
            f'inventory={self.inventory}, '
            f'num_scheduled={self.num_scheduled}, '
            f'inventory_target={self.inventory_target}, '
            f'should_schedule={self.predicate()}'
            ')'
        )


def trigger_generator(schedule_size_min: int, state: PoolInventoryState):
    while True:
        with state.cond:
            state.cond.wait_for(state.predicate)
            if state.abort:
                return

            schedule_size = max(
                schedule_size_min,
                state.inventory_target - state.inventory - state.num_scheduled,
            )
            logger.debug(f'state={state}, Need to schedule {schedule_size}.')
            state.num_scheduled += schedule_size
            for _ in range(schedule_size):
                yield None


class Pool(Generic[_T_CONFIG, _T_OUTPUT]):

    def __init__(self, config: PoolConfig[_T_CONFIG, _T_OUTPUT]):
        self.config = config

        process_counter = multiprocessing.Value('i')
        with process_counter.get_lock():
            process_counter.value = 0  # type: ignore

        self.mp_pool = multiprocessing.Pool(
            processes=self.config.num_processes,
            initializer=pool_worker_initializer,
            initargs=(self.config, process_counter),
        )

        self.state = PoolInventoryState(
            cond=threading.Condition(threading.Lock()),
            inventory=0,
            num_scheduled=0,
            inventory_target=self.config.inventory,
            abort=False,
        )

        self.mp_pool_iter = self.mp_pool.imap_unordered(
            pool_worker_runner,
            trigger_generator(
                schedule_size_min=round(
                    self.config.schedule_size_min_factor * self.config.num_processes
                ),
                state=self.state,
            ),
        )

        self.cleanup_flag = False
        atexit.register(self.cleanup)

    def cleanup(self):
        if not self.cleanup_flag:
            self.cleanup_flag = True

            with self.state.cond:
                self.state.abort = True
                self.state.cond.notify()

            self.mp_pool.close()
            self.mp_pool.terminate()
            time.sleep(1)

            for proc in self.mp_pool._pool:  # type: ignore
                if not psutil.pid_exists(proc.pid):
                    continue
                logger.warning(f'worker pid={proc.pid} still exists, killing...')
                proc = psutil.Process(proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()

            # For gc.
            self.__setattr__('mp_pool_iter', None)
            self.__setattr__('state', None)
            self.__setattr__('mp_pool', None)
            self.__setattr__('config', None)

            atexit.unregister(self.cleanup)

    def run(self):
        output: _T_OUTPUT = self.mp_pool_iter.next(timeout=self.config.timeout)

        # Update inventory.
        with self.state.cond:
            with self.mp_pool_iter._cond:  # type: ignore
                new_inventory = len(self.mp_pool_iter._items)  # type: ignore
            logger.debug(f'inventory: {self.state.inventory} -> {new_inventory}')

            # NOTE: We have just get one output, hence need to minus one.
            num_scheduled_delta = new_inventory - self.state.inventory + 1
            logger.debug(f'num_scheduled_delta: {num_scheduled_delta}')
            assert num_scheduled_delta >= 0

            new_num_scheduled = self.state.num_scheduled - num_scheduled_delta
            logger.debug(f'num_scheduled: {self.state.num_scheduled} -> {new_num_scheduled}')

            self.state.inventory = new_inventory
            self.state.num_scheduled = new_num_scheduled

            # Wake up trigger_generator.
            self.state.cond.notify()

        return output
