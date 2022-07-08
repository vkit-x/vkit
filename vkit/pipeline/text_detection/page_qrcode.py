from typing import Sequence, List
import string

import attrs
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.utility import rng_choice_with_size
from vkit.element import Mask, ScoreMap
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep

CV_PAYLOAD_TEXT_LENGTH_MAX = 150


@attrs.define
class PageQrcodeStepConfig:
    payload_text_length_min: int = 1
    payload_text_length_max: int = CV_PAYLOAD_TEXT_LENGTH_MAX
    alpha_min: float = 0.7
    alpha_max: float = 1.0


@attrs.define
class PageQrcodeStepOutput:
    height: int
    width: int
    qrcode_score_maps: Sequence[ScoreMap]


class PageQrcodeStep(
    PipelineStep[
        PageQrcodeStepConfig,
        PageQrcodeStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageQrcodeStepConfig):
        super().__init__(config)

        assert self.config.payload_text_length_max <= CV_PAYLOAD_TEXT_LENGTH_MAX
        self.ascii_letters = tuple(string.ascii_letters)

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_layout_step_output = state.get_pipeline_step_output(PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        qrcode_score_maps: List[ScoreMap] = []
        for layout_qrcode in page_layout.layout_qrcodes:
            box = layout_qrcode.box
            assert box.height == box.width

            payload_text_length = rng.integers(
                self.config.payload_text_length_min,
                self.config.payload_text_length_max + 1,
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
                rng.uniform(self.config.alpha_min, self.config.alpha_max)
            )

            if qrcode_score_map.height != box.height:
                qrcode_score_map = qrcode_score_map.to_resized_score_map(resized_height=box.height)

            qrcode_score_map = qrcode_score_map.to_box_attached(box)

            qrcode_score_maps.append(qrcode_score_map)

        return PageQrcodeStepOutput(
            height=page_layout.height,
            width=page_layout.width,
            qrcode_score_maps=qrcode_score_maps,
        )


page_qrcode_step_factory = PipelineStepFactory(PageQrcodeStep)
