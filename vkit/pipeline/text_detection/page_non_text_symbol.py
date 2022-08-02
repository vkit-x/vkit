from typing import Sequence, List, Union
from enum import Enum, unique

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np

from vkit.utility import normalize_to_keys_and_probs, rng_choice
from vkit.element import Box, Image, ImageKind
from vkit.engine.image import selector_image_factory
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep


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
        PageNonTextSymbolStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageNonTextSymbolStepConfig):
        super().__init__(config)

        self.symbol_image_selector = selector_image_factory.create({
            'image_folders': self.config.symbol_image_folders,
            'target_kind_image': None,
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

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_layout_step_output = state.get_pipeline_step_output(PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        images: List[Image] = []
        boxes: Sequence[Box] = []
        alphas: List[Union[np.ndarray, float]] = []

        for layout_non_text_symbol in page_layout.layout_non_text_symbols:
            box = layout_non_text_symbol.box

            image = self.symbol_image_selector.run(
                {
                    'height': box.height,
                    'width': box.width
                },
                rng,
            )
            alpha: Union[np.ndarray, float] = layout_non_text_symbol.alpha

            if image.kind == ImageKind.RGBA:
                # Extract and rescale alpha.
                np_alpha = (image.mat[:, :, 3]).astype(np.float32) / 255
                np_alpha_max = np_alpha.max()
                np_alpha *= layout_non_text_symbol.alpha
                np_alpha /= np_alpha_max
                alpha = np_alpha

                # Force to rgb (ignoring alpha channel).
                image = Image(mat=image.mat[:, :, :3])

            elif image.kind == ImageKind.GRAYSCALE:
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
