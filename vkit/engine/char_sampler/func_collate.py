from typing import Sequence, List

from numpy.random import Generator as RandomGenerator

from vkit.engine.interface import EngineExecutorAggregatorSelector
from .type import CharSamplerEngineRunConfig


def char_sampler_func_collate(
    selector: EngineExecutorAggregatorSelector[
        CharSamplerEngineRunConfig,
        Sequence[str],
    ],
    run_config: CharSamplerEngineRunConfig,
    rng: RandomGenerator,
):  # yapf: disable

    if run_config.enable_aggregator_mode:
        num_chars = run_config.num_chars

        chars: List[str] = []
        while len(chars) < num_chars:
            if chars and rng.random() < 0.5:
                chars.append(' ')
            new_chars = selector.select_engine_executor(rng).run(run_config, rng)
            chars.extend(new_chars)

        # Trim and make sure the last char is not space.
        if len(chars) > num_chars:
            rest = chars[num_chars:]
            chars = chars[:num_chars]
            if chars[-1].isspace():
                chars.pop()
                assert not rest[0].isspace()
                chars.append(rest[0])

        assert len(chars) == num_chars
        return chars

    else:
        return selector.select_engine_executor(rng).run(run_config, rng)
