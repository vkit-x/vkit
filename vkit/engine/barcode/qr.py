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
from typing import Optional
import string

import attrs
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.utility import rng_choice_with_size
from vkit.element import Mask, ScoreMap
from vkit.engine.interface import (
    Engine,
    EngineExecutorFactory,
    NoneTypeEngineInitResource,
)
from .type import BarcodeEngineRunConfig

CV_PAYLOAD_TEXT_LENGTH_MAX = 150


@attrs.define
class BarcodeQrEngineInitConfig:
    payload_text_length_min: int = 1
    payload_text_length_max: int = CV_PAYLOAD_TEXT_LENGTH_MAX
    alpha_min: float = 0.7
    alpha_max: float = 1.0


class BarcodeQrEngine(
    Engine[
        BarcodeQrEngineInitConfig,
        NoneTypeEngineInitResource,
        BarcodeEngineRunConfig,
        ScoreMap,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'qr'

    def __init__(
        self,
        init_config: BarcodeQrEngineInitConfig,
        init_resource: Optional[NoneTypeEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        assert self.init_config.payload_text_length_max <= CV_PAYLOAD_TEXT_LENGTH_MAX
        self.ascii_letters = tuple(string.ascii_letters)

    def run(self, run_config: BarcodeEngineRunConfig, rng: RandomGenerator) -> ScoreMap:
        payload_text_length = rng.integers(
            self.init_config.payload_text_length_min,
            self.init_config.payload_text_length_max + 1,
        )
        payload_text = ''.join(
            rng_choice_with_size(rng, self.ascii_letters, size=payload_text_length)
        )

        qrcode_encoder = cv.QRCodeEncoder.create()
        # Black as activated pixels.
        mask = Mask(mat=qrcode_encoder.encode(payload_text)).to_inverted_mask()
        assert mask.height == mask.width

        qrcode_score_map = ScoreMap.from_shapable(mask)
        qrcode_score_map[mask] = float(
            rng.uniform(self.init_config.alpha_min, self.init_config.alpha_max)
        )

        if qrcode_score_map.shape != (run_config.height, run_config.width):
            qrcode_score_map = qrcode_score_map.to_resized_score_map(
                resized_height=run_config.height,
                resized_width=run_config.width,
            )

        return qrcode_score_map


barcode_qr_engine_executor_factory = EngineExecutorFactory(BarcodeQrEngine)
