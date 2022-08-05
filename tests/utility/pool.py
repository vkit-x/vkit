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
