'''
Consts for detecting delimiter chars.
'''
from typing import Sequence, Tuple

#: Delimiters.
ITV_DELIMITER: Sequence[Sequence[Tuple[int, int]]] = [
    # ASCII_DELIMITERS_RANGES
    [
        (0x0021, 0x002F),
        (0x003A, 0x0040),
        (0x005B, 0x0060),
        (0x007B, 0x007E),
        # ¢, £, ¤, ¥
        (0x00A2, 0x00A5),
    ],
    [
        # Pick from the whitespace category.
        (0xB7, 0xB7)
    ],

    # GENERAL_DELIMITERS_RAGES
    # http://www.unicode.org/charts/PDF/U2000.pdf
    [
        # (0x2000, 0x206F),
        # Fix with:
        (0x2010, 0x2027),
        (0x202D, 0x202E),
        (0x2030, 0x205E),
    ],
    # CJK_DELIMITERS_RANGES
    # http://www.unicode.org/charts/PDF/U3000.pdf
    # http://www.unicode.org/charts/PDF/UFE30.pdf
    [
        # (0x3000, 0x303F),
        # Fix with:
        (0x3001, 0x3006),
        (0x3008, 0x303F),
        (0xFE30, 0xFE4F),
    ],
    # DELIMITERS_EXTENSION_RANGES
    # http://www.unicode.org/charts/PDF/UFF00.pdf
    [
        (0xFF01, 0xFF0F),
        (0xFF1A, 0xFF20),
        (0xFF3B, 0xFF40),
        (0xFF5B, 0xFF64),
        (0xFFE0, 0xFFEE),
    ],
]

DELIMITER_BLACKLIST = {
    '々',
    '〓',
    "〒",
    '〆',
}
