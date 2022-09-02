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
import logging
import time

from numpy.random import SeedSequence, default_rng
import attrs

from vkit.utility.pool import Pool, PoolConfig


@attrs.define
class PoolWorkerDemoConfig:
    pass


class PoolWorkerDemo:

    def __init__(
        self,
        process_idx: int,
        seed_sequence: SeedSequence,
        logger: logging.Logger,
        config: PoolWorkerDemoConfig,
    ):
        self.process_idx = process_idx
        self.rng = default_rng(seed_sequence)
        logger.info(self.rng.bit_generator.state)
        self.logger = logger

    def run(self):
        msg = f'{self.process_idx}-{self.rng.integers(0, 65536)}'
        sleep = self.rng.uniform(0.0, 1.0)
        self.logger.info(f'{msg}, sleep={sleep}')
        time.sleep(sleep)
        self.logger.info(f'{msg}, done sleep.')
        return msg


def debug_pool():
    pool = Pool(
        PoolConfig(
            inventory=30,
            num_processes=3,
            pool_worker_class=PoolWorkerDemo,
            pool_worker_config=PoolWorkerDemoConfig(),
            logging_level=logging.DEBUG,
        )
    )
    while True:
        print(pool.run())
        time.sleep(1.0)


def debug_pool_cleanup():
    import os
    print(os.getpid())
    breakpoint()

    pool = Pool(
        PoolConfig(
            inventory=30,
            num_processes=3,
            pool_worker_class=PoolWorkerDemo,
            pool_worker_config=PoolWorkerDemoConfig(),
            logging_level=logging.DEBUG,
        )
    )
    print('pool setup.')
    breakpoint()

    pool.cleanup()
    print('pool cleanup.')
    breakpoint()

    del pool
    import gc
    gc.collect()
    print('pool gc.')
    breakpoint()

    pool = Pool(
        PoolConfig(
            inventory=30,
            num_processes=3,
            pool_worker_class=PoolWorkerDemo,
            pool_worker_config=PoolWorkerDemoConfig(),
            logging_level=logging.DEBUG,
        )
    )
    print('pool setup again.')
    breakpoint()

    pool.cleanup()
    print('pool cleanup again.')
    breakpoint()

    del pool
    gc.collect()
    print('pool gc again.')
    breakpoint()
