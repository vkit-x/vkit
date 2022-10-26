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
from typing import List, Optional, Sequence
from os import PathLike

import attrs
from numpy.random import Generator as RandomGenerator
import iolite as io

from vkit.utility import rng_choice
from vkit.element import Image, ImageMode, Box
from vkit.engine.interface import (
    Engine,
    EngineExecutorFactory,
    NoneTypeEngineInitResource,
)
from .type import ImageEngineRunConfig


@attrs.define
class ImageSelectorEngineInitConfig:
    image_folders: Sequence[str]
    target_image_mode: Optional[ImageMode] = ImageMode.RGB
    force_resize: bool = False


class ImageSelectorEngine(
    Engine[
        ImageSelectorEngineInitConfig,
        NoneTypeEngineInitResource,
        ImageEngineRunConfig,
        Image,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'selector'

    def __init__(
        self,
        init_config: ImageSelectorEngineInitConfig,
        init_resource: Optional[NoneTypeEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        self.image_files: List[PathLike] = []
        for image_folder in self.init_config.image_folders:
            image_fd = io.folder(image_folder, expandvars=True, exists=True)
            for ext in ['jpg', 'jpeg', 'png']:
                for new_ext in [ext, ext.upper()]:
                    self.image_files.extend(image_fd.glob(f'**/*.{new_ext}'))

    def run(self, run_config: ImageEngineRunConfig, rng: RandomGenerator) -> Image:
        image_file = rng_choice(rng, self.image_files)
        image = Image.from_file(image_file)

        if self.init_config.target_image_mode:
            image = image.to_target_mode_image(self.init_config.target_image_mode)

        if run_config.disable_resizing:
            assert run_config.height == 0 and run_config.width == 0
            return image

        height = run_config.height
        width = run_config.width
        if not self.init_config.force_resize and height <= image.height and width <= image.width:
            # Select a part of image.
            up = rng.integers(0, image.height - height + 1)
            left = rng.integers(0, image.width - width + 1)
            box = Box(
                up=up,
                down=up + height - 1,
                left=left,
                right=left + width - 1,
            )
            image = box.extract_image(image)

        else:
            # Resize image.
            image = image.to_resized_image(
                resized_height=height,
                resized_width=width,
            )

        return image


image_selector_engine_executor_factory = EngineExecutorFactory(ImageSelectorEngine)
