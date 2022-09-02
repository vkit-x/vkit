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
from typing import Sequence, List, Mapping, Any, Optional

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import ScoreMap
from vkit.engine.barcode import (
    barcode_qr_engine_executor_factory,
    barcode_code39_engine_executor_factory,
)

from ..interface import PipelineStep, PipelineStepFactory
from .page_layout import PageLayoutStepOutput


@attrs.define
class PageBarcodeStepConfig:
    barcode_qr_config: Optional[Mapping[str, Any]] = None
    barcode_code39_config: Optional[Mapping[str, Any]] = None


@attrs.define
class PageBarcodeStepInput:
    page_layout_step_output: PageLayoutStepOutput


@attrs.define
class PageBarcodeStepOutput:
    height: int
    width: int
    barcode_qr_score_maps: Sequence[ScoreMap]
    barcode_code39_score_maps: Sequence[ScoreMap]


class PageBarcodeStep(
    PipelineStep[
        PageBarcodeStepConfig,
        PageBarcodeStepInput,
        PageBarcodeStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageBarcodeStepConfig):
        super().__init__(config)

        self.barcode_qr_engine_executor = barcode_qr_engine_executor_factory.create(
            self.config.barcode_qr_config
        )
        self.barcode_code39_engine_executor = barcode_code39_engine_executor_factory.create(
            self.config.barcode_code39_config
        )

    def run(self, input: PageBarcodeStepInput, rng: RandomGenerator):
        page_layout_step_output = input.page_layout_step_output
        page_layout = page_layout_step_output.page_layout

        barcode_qr_score_maps: List[ScoreMap] = []
        for layout_barcode_qr in page_layout.layout_barcode_qrs:
            box = layout_barcode_qr.box
            assert box.height == box.width

            barcode_qr_score_map = self.barcode_qr_engine_executor.run(
                {
                    'height': box.height,
                    'width': box.width,
                },
                rng=rng,
            )
            barcode_qr_score_map = barcode_qr_score_map.to_box_attached(box)
            barcode_qr_score_maps.append(barcode_qr_score_map)

        barcode_code39_score_maps: List[ScoreMap] = []
        for layout_barcode_code39 in page_layout.layout_barcode_code39s:
            box = layout_barcode_code39.box

            barcode_code39_score_map = self.barcode_code39_engine_executor.run(
                {
                    'height': box.height,
                    'width': box.width,
                },
                rng=rng,
            )
            barcode_code39_score_map = barcode_code39_score_map.to_box_attached(box)
            barcode_code39_score_maps.append(barcode_code39_score_map)

        return PageBarcodeStepOutput(
            height=page_layout.height,
            width=page_layout.width,
            barcode_qr_score_maps=barcode_qr_score_maps,
            barcode_code39_score_maps=barcode_code39_score_maps,
        )


page_barcode_step_factory = PipelineStepFactory(PageBarcodeStep)
