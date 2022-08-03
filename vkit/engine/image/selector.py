from typing import List, Optional, Sequence
from os import PathLike

import attrs
from numpy.random import Generator as RandomGenerator
import iolite as io

from vkit.utility import rng_choice
from vkit.element import Image, ImageKind, Box
from vkit.engine.interface import Engine, NoneTypeEngineResource
from .type import ImageEngineRunConfig


@attrs.define
class SelectorImageEngineConfig:
    image_folders: Sequence[str]
    target_kind_image: Optional[ImageKind] = ImageKind.RGB
    force_resize: bool = False


class SelectorImageEngine(
    Engine[
        SelectorImageEngineConfig,
        NoneTypeEngineResource,
        ImageEngineRunConfig,
        Image,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'selector'

    def __init__(
        self,
        config: SelectorImageEngineConfig,
        resource: Optional[NoneTypeEngineResource] = None,
    ):
        super().__init__(config, resource)

        self.image_files: List[PathLike] = []
        for image_folder in self.config.image_folders:
            image_fd = io.folder(image_folder, expandvars=True, exists=True)
            for ext in ['jpg', 'jpeg', 'png']:
                for new_ext in [ext, ext.upper()]:
                    self.image_files.extend(image_fd.glob(f'**/*.{new_ext}'))

    def run(self, run_config: ImageEngineRunConfig, rng: RandomGenerator) -> Image:
        image_file = rng_choice(rng, self.image_files)
        image = Image.from_file(image_file)

        if self.config.target_kind_image:
            image = image.to_target_kind_image(self.config.target_kind_image)

        height = run_config.height
        width = run_config.width
        if not self.config.force_resize and height <= image.height and width <= image.width:
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
