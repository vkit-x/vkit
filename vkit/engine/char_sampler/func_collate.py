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
