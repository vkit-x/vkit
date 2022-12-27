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
from typing import List, Tuple
from enum import Enum, unique
from itertools import product


@unique
class CornerState(Enum):
    NOT_MOVE = 'not_move'
    MOVE_VERT = 'move_vert'
    MOVE_HORI = 'move_hori'
    MOVE_INSIDE = 'move_inside'


def check_corner_states_are_valid(
    up_left_state: CornerState,
    up_right_state: CornerState,
    down_right_state: CornerState,
    down_left_state: CornerState,
):
    # Check up & down border.
    deviated_states = (CornerState.MOVE_VERT, CornerState.MOVE_INSIDE)
    if up_left_state in deviated_states and up_right_state in deviated_states:
        return False
    if down_left_state in deviated_states and down_right_state in deviated_states:
        return False

    # Check left & right border.
    deviated_states = (CornerState.MOVE_HORI, CornerState.MOVE_INSIDE)
    if up_left_state in deviated_states and down_left_state in deviated_states:
        return False
    if up_right_state in deviated_states and down_right_state in deviated_states:
        return False

    return True


def generate_corner_states():
    valid_corner_states: List[Tuple[CornerState, CornerState, CornerState, CornerState]] = []
    for corner_states in product(CornerState, CornerState, CornerState, CornerState):
        text = (
            f'UL: {corner_states[0].value}, UR: {corner_states[1].value}, '
            f'DR: {corner_states[2].value}, DL: {corner_states[3].value}'
        )
        if check_corner_states_are_valid(*corner_states):
            valid_corner_states.append(corner_states)
            print('valid:', text)
        else:
            print('!!! invalid:', text)
            pass
    return valid_corner_states


def debug():
    valid_corner_states = generate_corner_states()
    print('#valid_corner_states', len(valid_corner_states))
    breakpoint()
