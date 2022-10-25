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
from typing import Sequence, List, Union
from enum import Enum, unique

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np

from vkit.utility import normalize_to_keys_and_probs, rng_choice
from vkit.element import Box, Image, ImageMode
from vkit.engine.image import image_selector_engine_executor_factory
from ..interface import PipelineStep, PipelineStepFactory
from .page_layout import PageLayoutStepOutput


@attrs.define
class PageNonTextSymbolStepConfig:
    symbol_image_folders: Sequence[str]

    weight_color_grayscale: float = 0.9
    color_grayscale_min: int = 0
    color_grayscale_max: int = 75
    weight_color_red: float = 0.04
    weight_color_green: float = 0.02
    weight_color_blue: float = 0.04
    color_rgb_min: int = 128
    color_rgb_max: int = 255


@attrs.define
class PageNonTextSymbolStepInput:
    page_layout_step_output: PageLayoutStepOutput


@attrs.define
class PageNonTextSymbolStepOutput:
    images: Sequence[Image]
    boxes: Sequence[Box]
    alphas: Sequence[Union[np.ndarray, float]]


@unique
class NonTextSymbolColorMode(Enum):
    GRAYSCALE = 'grayscale'
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


class PageNonTextSymbolStep(
    PipelineStep[
        PageNonTextSymbolStepConfig,
        PageNonTextSymbolStepInput,
        PageNonTextSymbolStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageNonTextSymbolStepConfig):
        super().__init__(config)

        self.symbol_image_selector_engine_executor = \
            image_selector_engine_executor_factory.create({
                'image_folders': self.config.symbol_image_folders,
                'target_image_mode': None,
                'force_resize': True,
            })

        self.color_modes, self.color_modes_probs = normalize_to_keys_and_probs([
            (
                NonTextSymbolColorMode.GRAYSCALE,
                self.config.weight_color_grayscale,
            ),
            (
                NonTextSymbolColorMode.RED,
                self.config.weight_color_red,
            ),
            (
                NonTextSymbolColorMode.GREEN,
                self.config.weight_color_green,
            ),
            (
                NonTextSymbolColorMode.BLUE,
                self.config.weight_color_blue,
            ),
        ])

    def run(self, input: PageNonTextSymbolStepInput, rng: RandomGenerator):
        page_layout_step_output = input.page_layout_step_output
        page_layout = page_layout_step_output.page_layout

        images: List[Image] = []
        boxes: Sequence[Box] = []
        alphas: List[Union[np.ndarray, float]] = []

        for layout_non_text_symbol in page_layout.layout_non_text_symbols:
            box = layout_non_text_symbol.box

            image = self.symbol_image_selector_engine_executor.run(
                {
                    'height': box.height,
                    'width': box.width
                },
                rng,
            )
            alpha: Union[np.ndarray, float] = layout_non_text_symbol.alpha

            if image.mode == ImageMode.RGBA:
                # Extract and rescale alpha.
                np_alpha = (image.mat[:, :, 3]).astype(np.float32) / 255
                np_alpha_max = np_alpha.max()
                np_alpha *= layout_non_text_symbol.alpha
                np_alpha /= np_alpha_max
                alpha = np_alpha

                # Force to rgb (ignoring alpha channel).
                image = Image(mat=image.mat[:, :, :3])

            elif image.mode == ImageMode.GRAYSCALE:
                # As mask.
                alpha = (image.mat > 0).astype(np.float32)
                alpha *= layout_non_text_symbol.alpha

                # Generate image with color.
                color_mode = rng_choice(rng, self.color_modes, probs=self.color_modes_probs)
                if color_mode == NonTextSymbolColorMode.GRAYSCALE:
                    grayscale_value = int(
                        rng.integers(
                            self.config.color_grayscale_min,
                            self.config.color_grayscale_max + 1,
                        )
                    )
                    symbol_color = (grayscale_value,) * 3

                else:
                    rgb_value = int(
                        rng.integers(
                            self.config.color_rgb_min,
                            self.config.color_rgb_max + 1,
                        )
                    )
                    if color_mode == NonTextSymbolColorMode.RED:
                        symbol_color = (rgb_value, 0, 0)
                    elif color_mode == NonTextSymbolColorMode.GREEN:
                        symbol_color = (0, rgb_value, 0)
                    elif color_mode == NonTextSymbolColorMode.BLUE:
                        symbol_color = (0, 0, rgb_value)
                    else:
                        raise NotImplementedError()

                image = Image.from_shapable(image, value=symbol_color)

            else:
                raise NotImplementedError()

            images.append(image)
            boxes.append(layout_non_text_symbol.box)
            alphas.append(alpha)

        return PageNonTextSymbolStepOutput(
            images=images,
            boxes=boxes,
            alphas=alphas,
        )


page_non_text_symbol_step_factory = PipelineStepFactory(PageNonTextSymbolStep)
