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
from typing import Sequence, List, Tuple

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import Box, ScoreMap
from vkit.engine.font import TextLine
from ..interface import PipelineStep, PipelineStepFactory
from .page_text_line import PageTextLineStepOutput


@attrs.define
class PageTextLineBoundingBoxStepConfig:
    prob_non_short_text_line: float = 0.05
    prob_short_text_line: float = 0.3
    offset_ratio_min: float = 0.1
    offset_ratio_max: float = 2.0
    border_thickness_ratio_min: float = 0.0
    border_thickness_ratio_max: float = 0.125
    border_thickness_min: int = 1
    alpha_min: float = 0.9
    alpha_max: float = 1.0


@attrs.define
class PageTextLineBoundingBoxStepInput:
    page_text_line_step_output: PageTextLineStepOutput


@attrs.define
class PageTextLineBoundingBoxStepOutput:
    score_maps: Sequence[ScoreMap]
    colors: Sequence[Tuple[int, int, int]]


class PageTextLineBoundingBoxStep(
    PipelineStep[
        PageTextLineBoundingBoxStepConfig,
        PageTextLineBoundingBoxStepInput,
        PageTextLineBoundingBoxStepOutput,
    ]
):  # yapf: disable

    def sample_offset(self, ref_char_height: int, rng: RandomGenerator):
        offset_ratio = rng.uniform(
            self.config.offset_ratio_min,
            self.config.offset_ratio_max,
        )
        return round(offset_ratio * ref_char_height)

    def sample_border_thickness(self, ref_char_height: int, rng: RandomGenerator):
        offset_ratio = rng.uniform(
            self.config.border_thickness_ratio_min,
            self.config.border_thickness_ratio_max,
        )
        return max(round(offset_ratio * ref_char_height), self.config.border_thickness_min)

    def sample_text_line_bounding_box(
        self,
        height: int,
        width: int,
        text_line: TextLine,
        rng: RandomGenerator,
    ):
        ref_char_height_max = max(
            char_glyph.ref_char_height for char_glyph in text_line.char_glyphs
        )

        # Sample shape.
        offset_up = self.sample_offset(ref_char_height_max, rng)
        offset_down = self.sample_offset(ref_char_height_max, rng)
        offset_left = self.sample_offset(ref_char_height_max, rng)
        offset_right = self.sample_offset(ref_char_height_max, rng)

        box_height = text_line.box.height + offset_up + offset_down
        box_width = text_line.box.width + offset_left + offset_right

        border_thickness = self.sample_border_thickness(ref_char_height_max, rng)
        alpha = float(rng.uniform(self.config.alpha_max, self.config.alpha_max))

        # Fill empty area.
        score_map = ScoreMap.from_shape((box_height, box_width), value=alpha)

        empty_box = Box(
            up=border_thickness,
            down=box_height - border_thickness - 1,
            left=border_thickness,
            right=box_width - border_thickness - 1,
        )
        assert empty_box.up < empty_box.down
        assert empty_box.left < empty_box.right
        empty_box.fill_score_map(score_map, 0.0)

        # Trim if out-of-boundary.
        page_box_up = text_line.box.up - offset_up
        page_box_down = text_line.box.down + offset_down
        page_box_left = text_line.box.left - offset_left
        page_box_right = text_line.box.right + offset_right

        trim_up_size = 0
        if page_box_up < 0:
            trim_up_size = abs(page_box_up)

        trim_down_size = 0
        if page_box_down >= height:
            trim_down_size = page_box_down - height + 1

        trim_left_size = 0
        if page_box_left < 0:
            trim_left_size = abs(page_box_left)

        trim_right_size = 0
        if page_box_right >= width:
            trim_right_size = page_box_right - width + 1

        if trim_up_size > 0 \
                or trim_down_size > 0 \
                or trim_left_size > 0 \
                or trim_right_size > 0:
            trim_box = Box(
                up=trim_up_size,
                down=box_height - 1 - trim_down_size,
                left=trim_left_size,
                right=box_width - 1 - trim_right_size,
            )
            score_map = trim_box.extract_score_map(score_map)

        page_box = Box(
            up=max(0, page_box_up),
            down=min(height - 1, page_box_down),
            left=max(0, page_box_left),
            right=min(width - 1, page_box_right),
        )
        score_map = score_map.to_box_attached(page_box)

        return score_map, text_line.glyph_color

    def run(self, input: PageTextLineBoundingBoxStepInput, rng: RandomGenerator):
        page_text_line_step_output = input.page_text_line_step_output
        page_text_line_collection = page_text_line_step_output.page_text_line_collection

        score_maps: List[ScoreMap] = []
        colors: List[Tuple[int, int, int]] = []

        for text_line, is_short_text_line in zip(
            page_text_line_collection.text_lines,
            page_text_line_collection.short_text_line_flags,
        ):
            add_text_line_bounding_box = False
            if is_short_text_line:
                if rng.random() < self.config.prob_short_text_line:
                    add_text_line_bounding_box = True
            else:
                if rng.random() < self.config.prob_non_short_text_line:
                    add_text_line_bounding_box = True
            if not add_text_line_bounding_box:
                continue

            # Assign a bounding box.
            score_map, color = self.sample_text_line_bounding_box(
                height=page_text_line_collection.height,
                width=page_text_line_collection.width,
                text_line=text_line,
                rng=rng,
            )
            score_maps.append(score_map)
            colors.append(color)

        return PageTextLineBoundingBoxStepOutput(
            score_maps=score_maps,
            colors=colors,
        )


page_text_line_bounding_box_step_factory = PipelineStepFactory(PageTextLineBoundingBoxStep)
