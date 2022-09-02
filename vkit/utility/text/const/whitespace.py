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
Consts for detecting whitespace chars.
'''
from typing import Sequence, Tuple

#: Whitespace Chars.
#: Pulled from https://en.wikipedia.org/wiki/Whitespace_character
#:
#: Table 1.
#:
#: 0x9
#: 0xA
#: 0xB
#: 0xC
#: 0xD
#: 0x20
#: 0x85
#: 0xA0
#: 0x1680
#: 0x2000
#: 0x2001
#: 0x2002
#: 0x2003
#: 0x2004
#: 0x2005
#: 0x2006
#: 0x2007
#: 0x2008
#: 0x2009
#: 0x200A
#: 0x2028
#: 0x2029
#: 0x202F
#: 0x205F
#: 0x3000
#:
#: Table 2.
#:
#: 0x180E
#: 0x200B
#: 0x200C
#: 0x200D
#: 0x2060
#: 0xFEFF
#:
#: Table 3.
#:
#: 0xB7
#: 0x237D
#: 0x2420
#: 0x2422
#: 0x2423
#:
ITV_WHITESPACE: Sequence[Sequence[Tuple[int, int]]] = [[
    (0x9, 0xD),
    (0x20, 0x20),
    (0x85, 0x85),
    (0xA0, 0xA0),

    # Move the "middle dot" to delimiter category,
    # since this one is commonly used in Chinese news material.
    # (0xB7, 0xB7),
    (0x1680, 0x1680),
    (0x180E, 0x180E),

    # (0x2000, 0x200D),
    # Fix with:
    (0x2000, 0x200F),

    # (0x2028, 0x2029),
    # Fix with:
    (0x2028, 0x202C),
    (0x202F, 0x202F),

    # (0x205F, 0x2060),
    # Fix with:
    (0x205F, 0x206F),
    (0x237D, 0x237D),
    (0x2420, 0x2420),
    (0x2422, 0x2423),
    (0x3000, 0x3000),
    (0xFEFF, 0xFEFF),
]]
