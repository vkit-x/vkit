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
from typing import Sequence, Optional

import attrs

from vkit.element import Box, Polygon, Mask


@attrs.define
class CharMaskEngineRunConfig:
    height: int
    width: int
    char_polygons: Sequence[Polygon]
    char_bounding_boxes: Optional[Sequence[Box]] = None
    char_bounding_polygons: Optional[Sequence[Polygon]] = None


@attrs.define
class CharMask:
    combined_chars_mask: Mask
    char_masks: Optional[Sequence[Mask]] = None
