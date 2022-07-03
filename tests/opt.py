from typing import Mapping, Any

import os
import inspect

import iolite as io

from vkit.element import Image
from vkit.utility import get_data_folder


def read_image(rel_path: str):
    dataset_fd = io.folder(str(os.getenv('VKIT_DATASET')), exists=True)
    return Image.from_file(dataset_fd / rel_path)


def get_test_output_path(rel_path: str, frames_offset: int = 0):
    if not os.getenv('VKIT_ROOT') or not os.getenv('VKIT_DATA'):
        return

    frames = inspect.stack()
    frames_offset += 2
    module_path = frames[frames_offset].filename
    function_name = frames[frames_offset].function
    module_fd = io.folder(get_data_folder(module_path))
    test_fd = io.folder(module_fd / function_name, touch=True)
    test_output_path = test_fd / rel_path
    io.folder(test_output_path.parent, touch=True)
    return test_output_path


def write_image(rel_path: str, image: Image, frames_offset: int = 0):
    test_output_path = get_test_output_path(rel_path, frames_offset=frames_offset)
    if test_output_path:
        image.to_file(test_output_path)


def write_json(rel_path: str, data: Mapping[str, Any], frames_offset: int = 0):
    test_output_path = get_test_output_path(rel_path, frames_offset=frames_offset)
    if test_output_path:
        io.write_json(test_output_path, data, indent=2, ensure_ascii=False)
