from typing import Sequence, List, Any
import string

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
from barcode import Code39
from barcode.writer import BaseWriter, mm2px
from PIL import Image as PilImage, ImageDraw as PilImageDraw

from vkit.element import Mask, ScoreMap
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep


@attrs.define
class PageBarcodeStepConfig:
    aspect_ratio: float = 0.2854396602149411
    alpha_min: float = 0.7
    alpha_max: float = 1.0


@attrs.define
class PageBarcodeStepOutput:
    height: int
    width: int
    barcode_score_maps: Sequence[ScoreMap]


# REFERENCE: https://github.com/WhyNotHugo/python-barcode/blob/main/barcode/writer.py
class NoTextImageWriter(BaseWriter):  # type: ignore

    def __init__(self, mode: str = "L"):
        super().__init__(
            self._init,
            self._paint_module,
            None,
            self._finish,
        )
        self.mode = mode
        self.dpi = 300
        self._image = None
        self._draw = None

    def calculate_size(self, modules_per_line: int, number_of_lines: int):
        width = 2 * self.quiet_zone + modules_per_line * self.module_width
        height = 2.0 + self.module_height * number_of_lines
        return width, height

    def _init(self, code: str):
        width, height = self.calculate_size(len(code[0]), len(code))
        size = (int(mm2px(width, self.dpi)), int(mm2px(height, self.dpi)))
        self._image = PilImage.new(self.mode, size, self.background)  # type: ignore
        self._draw = PilImageDraw.Draw(self._image)  # type: ignore

    def _paint_module(self, xpos: int, ypos: int, width: int, color: Any):
        size = [
            (mm2px(xpos, self.dpi), mm2px(ypos, self.dpi)),
            (
                mm2px(xpos + width, self.dpi),
                mm2px(ypos + self.module_height, self.dpi),
            ),
        ]
        self._draw.rectangle(size, outline=color, fill=color)  # type: ignore

    def _finish(self):
        return self._image


class PageBarcodeStep(
    PipelineStep[
        PageBarcodeStepConfig,
        PageBarcodeStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageBarcodeStepConfig):
        super().__init__(config)

        self.ascii_letters = tuple(string.ascii_letters)

    @staticmethod
    def convert_barcode_pil_image_to_mask(barcode_pil_image: PilImage.Image):
        mat = np.array(barcode_pil_image)
        mask = Mask(mat=mat).to_inverted_mask()

        # Trim.
        np_hori_max = np.amax(mask.mat, axis=0)
        np_hori_nonzero = np.nonzero(np_hori_max)[0]
        assert len(np_hori_nonzero) >= 2
        left = np_hori_nonzero[0]
        right = np_hori_nonzero[-1]

        np_vert_max = np.amax(mask.mat, axis=1)
        np_vert_nonzero = np.nonzero(np_vert_max)[0]
        assert len(np_vert_nonzero) >= 2
        up = np_vert_nonzero[0]
        down = np_vert_nonzero[-1]

        mask_mat = mask.mat[up:down + 1, left:right + 1]
        mask = Mask(mask_mat)

        return mask

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_layout_step_output = state.get_pipeline_step_output(PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        barcode_score_maps: List[ScoreMap] = []
        for layout_barcode in page_layout.layout_barcodes:
            box = layout_barcode.box

            num_chars = max(1, round(box.width / (box.height * self.config.aspect_ratio)))
            text = ''.join(rng.choice(self.ascii_letters) for _ in range(num_chars))
            pil_image = Code39(code=text, writer=NoTextImageWriter()).render()

            mask = self.convert_barcode_pil_image_to_mask(pil_image)

            barcode_score_map = ScoreMap.from_shapable(mask)
            barcode_score_map[mask] = float(
                rng.uniform(self.config.alpha_min, self.config.alpha_max)
            )

            if barcode_score_map.shape != box.shape:
                barcode_score_map = barcode_score_map.to_resized_score_map(
                    resized_height=box.height,
                    resized_width=box.width,
                )

            barcode_score_map = barcode_score_map.to_box_attached(box)
            barcode_score_maps.append(barcode_score_map)

        return PageBarcodeStepOutput(
            height=page_layout.height,
            width=page_layout.width,
            barcode_score_maps=barcode_score_maps,
        )


page_barcode_step_factory = PipelineStepFactory(PageBarcodeStep)
