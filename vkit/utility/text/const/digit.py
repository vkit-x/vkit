'''
Consts for detecting digit chars.
'''
from typing import Sequence, Tuple

#: Digits.
ITV_DIGIT: Sequence[Sequence[Tuple[int, int]]] = [
    # ASCII_DIGIT_RANGES
    [
        (0x0030, 0x0039),
    ],
    # DIGIT_EXTENSION_RANGES
    [
        (0xFF10, 0xFF19),
    ],
    # CIRCLE DIGIT RANGES
    [
        # ① - ⑨
        (0x2460, 0x2468),
    ],
]
