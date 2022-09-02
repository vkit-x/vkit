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
