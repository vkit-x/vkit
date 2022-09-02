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
Consts for detecting chinese chars.
'''
from typing import Sequence, Tuple

#: English Chars.
ITV_ENGLISH: Sequence[Sequence[Tuple[int, int]]] = [
    # ASCII_ALPHA_RANGES
    [
        (0x0041, 0x005A),
        (0x0061, 0x007A),
    ],
    # ALPHA_EXTENSION_RANGES
    [
        (0xFF21, 0xFF3A),
        (0xFF41, 0xFF5A),
    ],
]
