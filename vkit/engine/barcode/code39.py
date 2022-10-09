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
from typing import Optional, Any
import string

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
from barcode import Code39
from barcode.writer import BaseWriter, mm2px
from PIL import Image as PilImage, ImageDraw as PilImageDraw

from vkit.element import Mask, ScoreMap
from vkit.engine.interface import (
    Engine,
    EngineExecutorFactory,
    NoneTypeEngineInitResource,
)
from .type import BarcodeEngineRunConfig


@attrs.define
class BarcodeCode39EngineInitConfig:
    aspect_ratio: float = 0.2854396602149411
    alpha_min: float = 0.7
    alpha_max: float = 1.0


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


class BarcodeCode39Engine(
    Engine[
        BarcodeCode39EngineInitConfig,
        NoneTypeEngineInitResource,
        BarcodeEngineRunConfig,
        ScoreMap,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'code39'

    @classmethod
    def convert_barcode_pil_image_to_mask(cls, barcode_pil_image: PilImage.Image):
        mat = np.asarray(barcode_pil_image)
        mask = Mask(mat=mat).to_inverted_mask()

        # Trim.
        np_hori_max = np.amax(mask.mat, axis=0)
        np_hori_nonzero = np.nonzero(np_hori_max)[0]
        assert len(np_hori_nonzero) >= 2
        left = int(np_hori_nonzero[0])
        right = int(np_hori_nonzero[-1])

        np_vert_max = np.amax(mask.mat, axis=1)
        np_vert_nonzero = np.nonzero(np_vert_max)[0]
        assert len(np_vert_nonzero) >= 2
        up = int(np_vert_nonzero[0])
        down = int(np_vert_nonzero[-1])

        mask = mask.to_cropped_mask(up=up, down=down, left=left, right=right)
        return mask

    def __init__(
        self,
        init_config: BarcodeCode39EngineInitConfig,
        init_resource: Optional[NoneTypeEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        self.ascii_letters = tuple(string.ascii_letters)

    def run(self, run_config: BarcodeEngineRunConfig, rng: RandomGenerator) -> ScoreMap:
        num_chars = max(
            1,
            round(run_config.width / (run_config.height * self.init_config.aspect_ratio)),
        )
        text = ''.join(rng.choice(self.ascii_letters) for _ in range(num_chars))
        pil_image = Code39(code=text, writer=NoTextImageWriter()).render()

        mask = self.convert_barcode_pil_image_to_mask(pil_image)

        barcode_score_map = ScoreMap.from_shapable(mask)
        barcode_score_map[mask] = float(
            rng.uniform(self.init_config.alpha_min, self.init_config.alpha_max)
        )

        if barcode_score_map.shape != (run_config.height, run_config.width):
            barcode_score_map = barcode_score_map.to_resized_score_map(
                resized_height=run_config.height,
                resized_width=run_config.width,
            )

        return barcode_score_map


barcode_code39_engine_executor_factory = EngineExecutorFactory(BarcodeCode39Engine)
