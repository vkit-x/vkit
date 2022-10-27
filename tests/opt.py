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
from typing import Mapping, Any
import subprocess
from pathlib import Path
import hashlib

import os
import inspect

import iolite as io

from vkit.element import Image
from vkit.utility import PathType


def read_image(rel_path: str):
    dataset_fd = io.folder(str(os.getenv('VKIT_DATASET')), exists=True)
    return Image.from_file(dataset_fd / rel_path)


def get_data_folder(file: PathType):
    proc = subprocess.run(
        f'$VKIT_ROOT/.direnv/bin/pyproject-data-folder "$VKIT_ROOT" "$VKIT_DATA" "{file}"',
        shell=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    data_folder = proc.stdout.strip()
    assert data_folder

    io.folder(data_folder, touch=True)

    return data_folder


def get_test_module_path_and_function_name(frames_offset: int):
    frames = inspect.stack()
    frames_offset += 3
    test_module_path = frames[frames_offset].filename
    test_function_name = frames[frames_offset].function
    return test_module_path, test_function_name


def get_test_output_path(rel_path: str, frames_offset: int):
    assert os.getenv('VKIT_ROOT') and os.getenv('VKIT_DATA')

    test_module_path, test_function_name = get_test_module_path_and_function_name(frames_offset)
    test_module_fd = io.folder(get_data_folder(test_module_path))
    test_function_name_fd = io.folder(test_module_fd / test_function_name, touch=True)
    test_output_path = test_function_name_fd / rel_path
    io.folder(test_output_path.parent, touch=True)
    return test_output_path


def get_test_approval_hash_file(rel_path: str, frames_offset: int):
    assert os.getenv('VKIT_ROOT') and os.getenv('VKIT_DATA')

    test_module_path, test_function_name = get_test_module_path_and_function_name(frames_offset)
    test_module_file = io.file(test_module_path, exists=True)
    tests_fd = io.folder('$VKIT_ROOT/tests', expandvars=True, exists=True)

    test_approval_hash_root_fd = io.folder(tests_fd / '.approval-hash')
    test_approval_hash_fd = io.folder(
        test_approval_hash_root_fd / test_module_file.relative_to(tests_fd) / test_function_name,
        touch=True,
    )
    test_approval_hash_file = test_approval_hash_fd / rel_path
    test_approval_hash_file = test_approval_hash_file.parent / (
        test_approval_hash_file.name + '.txt'
    )
    io.folder(test_approval_hash_file.parent, touch=True)
    return test_approval_hash_file


def check_or_update_approval_hash(test_output_path: Path, test_approval_hash_file: Path):
    assert test_output_path.exists()
    hash_algo = hashlib.sha256()
    hash_algo.update(test_output_path.read_bytes())
    test_output_hash = hash_algo.hexdigest()

    if not test_approval_hash_file.exists():
        test_approval_hash_file.write_text(test_output_hash)
    else:
        approval_hash = test_approval_hash_file.read_text()
        if approval_hash != test_output_hash:
            if os.getenv('VKIT_UPDATE_APPROVAL_HASH'):
                test_approval_hash_file.write_text(test_output_hash)
            else:
                raise RuntimeError('approval hash conflict.')


def write_image(rel_path: str, image: Image, frames_offset: int = 0):
    test_output_path = get_test_output_path(rel_path, frames_offset=frames_offset)
    if test_output_path:
        image.to_file(test_output_path)
        test_approval_hash_file = get_test_approval_hash_file(rel_path, frames_offset=frames_offset)
        check_or_update_approval_hash(test_output_path, test_approval_hash_file)


def write_json(rel_path: str, data: Mapping[str, Any], frames_offset: int = 0):
    test_output_path = get_test_output_path(rel_path, frames_offset=frames_offset)
    if test_output_path:
        io.write_json(test_output_path, data, indent=2, ensure_ascii=False)
        test_approval_hash_file = get_test_approval_hash_file(rel_path, frames_offset=frames_offset)
        check_or_update_approval_hash(test_output_path, test_approval_hash_file)
