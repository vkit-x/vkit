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
from vkit.element import Box, BoxOverlappingValidator


def test_box_overlapping_validator():
    box0 = Box(up=0, down=100, left=0, right=100)
    box_overlapping_validator = BoxOverlappingValidator([box0])
    assert box_overlapping_validator.is_overlapped(Box(up=100, down=100, left=100, right=100))
    assert not box_overlapping_validator.is_overlapped(Box(up=101, down=101, left=100, right=100))
    assert not box_overlapping_validator.is_overlapped(Box(up=100, down=100, left=101, right=101))
    assert not box_overlapping_validator.is_overlapped(Box(up=101, down=101, left=101, right=101))
