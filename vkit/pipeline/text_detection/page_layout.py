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
from typing import Optional, Sequence, List, DefaultDict
import math
import heapq
from enum import Enum, unique
import itertools
from collections import defaultdict

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import rng_choice, normalize_to_probs, normalize_to_keys_and_probs
from vkit.element import Box, BoxOverlappingValidator, Polygon
from vkit.engine.font import FontEngineRunConfigGlyphSequence
from .page_shape import PageShapeStepOutput
from ..interface import PipelineStep, PipelineStepFactory


@attrs.define
class PageLayoutStepConfig:
    # Text line heights.
    reference_aspect_ratio: float = 1 / 1.4142

    # Grid points.
    grid_pad_ratio_min: float = 0.01
    grid_pad_ratio_max: float = 0.05
    grid_step_ratio_min: float = 1.0
    grid_step_ratio_max: float = 1.1
    grid_vert_gap_ratio_min: float = 0.0
    grid_vert_gap_ratio_max: float = 0.5
    grid_hori_gap_ratio_min: float = 1.0
    grid_hori_gap_ratio_max: float = 1.15

    # Large text line.
    prob_add_large_text_line: float = 0.25
    large_text_line_height_ratio_min: float = 0.05
    large_text_line_height_ratio_max: float = 0.075
    large_text_line_length_ratio_min: float = 0.5
    large_text_line_length_ratio_max: float = 1.0

    # Normal text line.
    num_normal_text_line_heights_min: int = 2
    num_normal_text_line_heights_max: int = 4
    normal_text_line_height_ratio_min: float = 0.006
    normal_text_line_height_ratio_max: float = 0.036
    force_add_normal_text_line_height_ratio_min: bool = True

    # Non-text symbol.
    num_non_text_symbols_min: int = 0
    num_non_text_symbols_max: int = 5
    num_retries_to_get_non_overlapped_non_text_symbol: int = 5
    non_text_symbol_height_ratio_min: float = 0.018
    non_text_symbol_height_ratio_max: float = 0.064
    non_text_symbol_aspect_ratio_min: float = 0.9
    non_text_symbol_aspect_ratio_max: float = 1.111
    non_text_symbol_non_overlapped_alpha_min: float = 0.8
    non_text_symbol_non_overlapped_alpha_max: float = 1.0
    non_text_symbol_overlapped_alpha_min: float = 0.15
    non_text_symbol_overlapped_alpha_max: float = 0.55

    prob_normal_text_line_diff_heights_gap: float = 0.5
    prob_normal_text_line_gap: float = 0.5
    normal_text_line_gap_ratio_min: float = 0.05
    normal_text_line_gap_ratio_max: float = 1.25
    normal_text_line_length_ratio_min: float = 0.5
    normal_text_line_length_ratio_max: float = 1.0

    # Image.
    num_images_min: int = 0
    num_images_max: int = 3
    image_height_ratio_min: float = 0.1
    image_height_ratio_max: float = 0.35
    image_width_ratio_min: float = 0.1
    image_width_ratio_max: float = 0.35

    # Barcode (qr).
    num_barcode_qrs_min: int = 0
    num_barcode_qrs_max: int = 2
    barcode_qr_length_ratio_min: float = 0.05
    barcode_qr_length_ratio_max: float = 0.15

    # Barcode (code39).
    num_barcode_code39s_min: int = 0
    num_barcode_code39s_max: int = 2
    barcode_code39_height_ratio_min: float = 0.025
    barcode_code39_height_ratio_max: float = 0.05
    barcode_code39_aspect_ratio: float = 0.2854396602149411
    barcode_code39_num_chars_min: int = 9
    barcode_code39_num_chars_max: int = 13

    # Seal impression.
    num_seal_impressions_min: int = 1
    num_seal_impressions_max: int = 3
    seal_impression_angle_min: int = -45
    seal_impression_angle_max: int = 45
    seal_impression_height_ratio_min: float = 0.1
    seal_impression_height_ratio_max: float = 0.2
    seal_impression_weight_circle: float = 1
    seal_impression_weight_general_ellipse: float = 1
    seal_impression_general_ellipse_aspect_ratio_min: float = 0.75
    seal_impression_general_ellipse_aspect_ratio_max: float = 1.333

    # For char-level polygon regression.
    disconnected_text_region_polygons_height_ratio_max: float = 2.0


@attrs.define
class PageLayoutStepInput:
    page_shape_step_output: PageShapeStepOutput


@attrs.define
class LayoutTextLine:
    # grid_idx:
    #   == -1: for large text line.
    #   >= 0: for normal text lines.
    grid_idx: int
    # text_line_idx: index within a grid.
    text_line_idx: int
    text_line_height: int
    box: Box
    glyph_sequence: FontEngineRunConfigGlyphSequence


@attrs.define
class LayoutNonTextSymbol:
    box: Box
    alpha: float


@attrs.define
class LayoutSealImpression:
    box: Box
    angle: int


@attrs.define
class LayoutImage:
    box: Box


@attrs.define
class LayoutBarcodeQr:
    box: Box


@attrs.define
class LayoutBarcodeCode39:
    box: Box


@unique
class LayoutXcodePlacement(Enum):
    NEXT_TO_UP = 'next_to_up'
    NEXT_TO_DOWN = 'next_to_down'
    NEXT_TO_LEFT = 'next_to_left'
    NEXT_TO_RIGHT = 'next_to_right'


@attrs.define
class DisconnectedTextRegion:
    polygon: Polygon


@attrs.define
class NonTextRegion:
    polygon: Polygon


@unique
class LayoutNonTextLineDirection(Enum):
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'


@attrs.define
class PageLayout:
    height: int
    width: int
    layout_text_lines: Sequence[LayoutTextLine]
    layout_non_text_symbols: Sequence[LayoutNonTextSymbol]
    layout_seal_impressions: Sequence[LayoutSealImpression]
    layout_images: Sequence[LayoutImage]
    layout_barcode_qrs: Sequence[LayoutBarcodeQr]
    layout_barcode_code39s: Sequence[LayoutBarcodeCode39]
    disconnected_text_regions: Sequence[DisconnectedTextRegion]
    non_text_regions: Sequence[NonTextRegion]


@attrs.define
class PageLayoutStepOutput:
    page_layout: PageLayout
    debug_large_text_line_gird: Optional[Box]
    debug_grids: Sequence[Box]


@attrs.define(order=True)
class PrioritizedSegment:
    vert_begin_idx: int = attrs.field(order=True)
    hori_begin_idx: int = attrs.field(order=False)
    hori_end_idx: int = attrs.field(order=False)


@unique
class SealImpressionEllipseShapeMode(Enum):
    CIRCLE = 'circle'
    GENERAL_ELLIPSE = 'general_ellipse'


class PageLayoutStep(
    PipelineStep[
        PageLayoutStepConfig,
        PageLayoutStepInput,
        PageLayoutStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageLayoutStepConfig):
        super().__init__(config)

        (
            self.seal_impression_ellipse_shape_modes,
            self.seal_impression_ellipse_shape_modes_probs,
        ) = normalize_to_keys_and_probs([
            (
                SealImpressionEllipseShapeMode.CIRCLE,
                self.config.seal_impression_weight_circle,
            ),
            (
                SealImpressionEllipseShapeMode.GENERAL_ELLIPSE,
                self.config.seal_impression_weight_general_ellipse,
            ),
        ])

    def sample_large_text_line_height(self, reference_height: int, rng: RandomGenerator):
        if rng.random() < self.config.prob_add_large_text_line:
            large_text_line_height_ratio = rng.uniform(
                self.config.large_text_line_height_ratio_min,
                self.config.large_text_line_height_ratio_max,
            )
            return round(large_text_line_height_ratio * reference_height)

        else:
            return None

    def sample_normal_text_line_heights(self, reference_height: int, rng: RandomGenerator):
        normal_text_line_heights: List[int] = []

        if self.config.force_add_normal_text_line_height_ratio_min:
            normal_text_line_heights.append(
                round(self.config.normal_text_line_height_ratio_min * reference_height)
            )

        num_normal_text_line_heights = rng.integers(
            self.config.num_normal_text_line_heights_min,
            self.config.num_normal_text_line_heights_max + 1,
        )
        ratio_step = (
            self.config.normal_text_line_height_ratio_max
            - self.config.normal_text_line_height_ratio_min
        ) / num_normal_text_line_heights
        for step_idx in range(num_normal_text_line_heights):
            ratio_min = self.config.normal_text_line_height_ratio_min + step_idx * ratio_step
            ratio_max = ratio_min + ratio_step
            ratio = rng.uniform(ratio_min, ratio_max)
            normal_text_line_heights.append(round(ratio * reference_height))

        assert normal_text_line_heights
        return sorted(normal_text_line_heights)

    @classmethod
    def generate_grid_points(
        cls,
        grid_pad_ratio: float,
        grid_step: int,
        grid_gap: int,
        grid_gap_min: Optional[int],
        length: int,
        rng: RandomGenerator,
    ):
        grid_pad = min(length - grid_step, length * grid_pad_ratio)
        assert grid_pad > 0

        num_steps = (length - grid_pad + grid_gap) / (grid_step + grid_gap)
        if not num_steps.is_integer():
            num_steps = math.floor(num_steps)
        num_steps = int(num_steps)

        grid_pad = length - grid_step * num_steps - grid_gap * (num_steps - 1)
        assert grid_pad > 0
        grid_pad = grid_pad // 2

        begin = grid_pad
        end = grid_pad + grid_step - 1
        assert end < length - grid_pad

        begins: List[int] = []
        ends: List[int] = []

        while end < length - grid_pad:
            begins.append(begin)
            ends.append(end)

            cur_gap = grid_gap
            if grid_gap_min is not None:
                cur_gap = rng.integers(grid_gap_min, grid_gap + 1)

            begin = end + cur_gap
            end = begin + grid_step - 1

        return begins, ends

    def sample_grid_points(
        self,
        height: int,
        width: int,
        normal_text_line_heights_max: int,
        rng: RandomGenerator,
    ):
        grid_pad_ratio = rng.uniform(
            self.config.grid_pad_ratio_min,
            self.config.grid_pad_ratio_max,
        )

        grid_step_ratio = rng.uniform(
            self.config.grid_step_ratio_min,
            self.config.grid_step_ratio_max,
        )
        grid_step = round(normal_text_line_heights_max * grid_step_ratio)

        grid_vert_gap_min = round(
            normal_text_line_heights_max * self.config.grid_vert_gap_ratio_min
        )
        grid_vert_gap_max = round(
            normal_text_line_heights_max * self.config.grid_vert_gap_ratio_max
        )
        vert_begins, vert_ends = self.generate_grid_points(
            grid_pad_ratio=grid_pad_ratio,
            grid_step=grid_step,
            grid_gap=grid_vert_gap_max,
            grid_gap_min=grid_vert_gap_min,
            length=height,
            rng=rng,
        )

        grid_hori_gap_ratio = rng.uniform(
            self.config.grid_hori_gap_ratio_min,
            self.config.grid_hori_gap_ratio_max,
        )
        grid_hori_gap = round(normal_text_line_heights_max * grid_hori_gap_ratio)
        grid_hori_gap = max(normal_text_line_heights_max, grid_hori_gap)
        hori_begins, hori_ends = self.generate_grid_points(
            grid_pad_ratio=grid_pad_ratio,
            grid_step=grid_step,
            grid_gap=grid_hori_gap,
            grid_gap_min=None,
            length=width,
            rng=rng,
        )
        return (vert_begins, vert_ends), (hori_begins, hori_ends)

    def trim_grid_points_for_large_text_line(
        self,
        large_text_line_height: int,
        vert_begins: Sequence[int],
        vert_ends: Sequence[int],
        hori_begins_min: int,
        hori_ends_max: int,
    ):
        idx = 0
        while idx < len(vert_begins) \
                and vert_ends[idx] + 1 - vert_begins[0] < large_text_line_height:
            idx += 1

        if idx >= len(vert_begins) - 1:
            return None, 0

        large_text_line_gird = Box(
            up=vert_ends[idx] - large_text_line_height + 1,
            down=vert_ends[idx],
            left=hori_begins_min,
            right=hori_ends_max,
        )
        return large_text_line_gird, idx + 1

    def sample_grids(
        self,
        vert_begins: Sequence[int],
        vert_ends: Sequence[int],
        hori_begins: Sequence[int],
        hori_ends: Sequence[int],
        rng: RandomGenerator,
    ):
        num_vert_ends = len(vert_ends)
        assert num_vert_ends == len(vert_begins)

        num_hori_ends = len(hori_ends)
        assert num_hori_ends == len(hori_begins)

        priority_queue = [
            PrioritizedSegment(
                vert_begin_idx=0,
                hori_begin_idx=0,
                hori_end_idx=num_hori_ends - 1,
            )
        ]
        grids: List[Box] = []
        while priority_queue:
            cur_segment = heapq.heappop(priority_queue)

            # Deal with segments in the same level.
            same_vert_segments: List[PrioritizedSegment] = []
            while priority_queue \
                    and priority_queue[0].vert_begin_idx == cur_segment.vert_begin_idx:
                same_vert_segments.append(heapq.heappop(priority_queue))

            if same_vert_segments:
                # Rebuid segments.
                same_vert_segments.append(cur_segment)
                same_vert_segments = sorted(
                    same_vert_segments,
                    key=lambda segment: segment.hori_begin_idx,
                )

                rebuilt_segments: List[PrioritizedSegment] = []
                rebuilt_begin = 0
                while rebuilt_begin < len(same_vert_segments):
                    rebuilt_end = rebuilt_begin
                    while rebuilt_end + 1 < len(same_vert_segments) \
                            and (same_vert_segments[rebuilt_end + 1].hori_begin_idx
                                 == same_vert_segments[rebuilt_end].hori_end_idx + 1):
                        rebuilt_end += 1
                    rebuilt_segments.append(
                        PrioritizedSegment(
                            vert_begin_idx=cur_segment.vert_begin_idx,
                            hori_begin_idx=same_vert_segments[rebuilt_begin].hori_begin_idx,
                            hori_end_idx=same_vert_segments[rebuilt_end].hori_end_idx,
                        )
                    )
                    rebuilt_begin = rebuilt_end + 1

                # Re-pick the first segment.
                cur_segment = rebuilt_segments[0]
                for other_segment in rebuilt_segments[1:]:
                    heapq.heappush(priority_queue, other_segment)

            # Generate grids for the current segment.
            vert_begin_idx = cur_segment.vert_begin_idx

            hori_begin_idx = cur_segment.hori_begin_idx
            hori_end_idx = cur_segment.hori_end_idx
            while hori_begin_idx <= hori_end_idx:
                # Randomly generate grid.
                cur_vert_end_idx = rng.integers(vert_begin_idx, num_vert_ends)

                # Try to sample segment with length >= 2.
                if hori_end_idx + 1 - hori_begin_idx <= 3:
                    cur_hori_end_idx = hori_end_idx
                else:
                    cur_hori_end_idx = rng.integers(hori_begin_idx + 1, hori_end_idx + 1)

                grids.append(
                    Box(
                        up=vert_begins[vert_begin_idx],
                        down=vert_ends[cur_vert_end_idx],
                        left=hori_begins[hori_begin_idx],
                        right=hori_ends[cur_hori_end_idx],
                    )
                )
                next_vert_begin_idx = cur_vert_end_idx + 1
                if next_vert_begin_idx < num_vert_ends:
                    heapq.heappush(
                        priority_queue,
                        PrioritizedSegment(
                            vert_begin_idx=next_vert_begin_idx,
                            hori_begin_idx=hori_begin_idx,
                            hori_end_idx=cur_hori_end_idx,
                        ),
                    )

                hori_begin_idx = cur_hori_end_idx + 1

        return grids

    @classmethod
    def calculate_normal_text_line_heights_probs(
        cls,
        normal_text_line_heights_expected_probs: Sequence[float],
        normal_text_line_heights_acc_areas: List[int],
    ):
        if sum(normal_text_line_heights_acc_areas) == 0:
            normal_text_line_heights_cur_probs = [0.0] * len(normal_text_line_heights_acc_areas)
        else:
            normal_text_line_heights_cur_probs = normalize_to_probs(
                normal_text_line_heights_acc_areas
            )

        probs = normalize_to_probs([
            max(0.0, expected_prob - cur_prob) for cur_prob, expected_prob in zip(
                normal_text_line_heights_cur_probs,
                normal_text_line_heights_expected_probs,
            )
        ])
        return probs

    def fill_normal_text_lines_to_grid(
        self,
        normal_text_line_heights: Sequence[int],
        normal_text_line_heights_expected_probs: Sequence[float],
        normal_text_line_heights_acc_areas: List[int],
        grid_idx: int,
        grid: Box,
        rng: RandomGenerator,
    ):
        normal_text_line_heights_indices = list(range(len(normal_text_line_heights)))
        normal_text_line_heights_max = normal_text_line_heights[-1]

        layout_text_lines: List[LayoutTextLine] = []
        up = grid.up
        prev_text_line_height: Optional[int] = None

        while up + normal_text_line_heights_max - 1 <= grid.down:
            normal_text_line_heights_probs = self.calculate_normal_text_line_heights_probs(
                normal_text_line_heights_expected_probs=normal_text_line_heights_expected_probs,
                normal_text_line_heights_acc_areas=normal_text_line_heights_acc_areas,
            )
            normal_text_line_height_idx = rng_choice(
                rng=rng,
                items=normal_text_line_heights_indices,
                probs=normal_text_line_heights_probs,
            )
            normal_text_line_height = normal_text_line_heights[normal_text_line_height_idx]

            add_gap = False
            if prev_text_line_height:
                if prev_text_line_height != normal_text_line_height:
                    add_gap = (rng.random() < self.config.prob_normal_text_line_diff_heights_gap)
                else:
                    add_gap = (rng.random() < self.config.prob_normal_text_line_gap)
            if add_gap:
                gap_ratio = rng.uniform(
                    self.config.normal_text_line_gap_ratio_min,
                    self.config.normal_text_line_gap_ratio_max,
                )
                gap = round(gap_ratio * normal_text_line_height)
                gap = min(grid.down - (up + normal_text_line_height - 1), gap)
                up += gap
            down = up + normal_text_line_height - 1
            assert down <= grid.down

            length_ratio = rng.uniform(
                self.config.normal_text_line_length_ratio_min,
                self.config.normal_text_line_length_ratio_max,
            )
            normal_text_line_length = round(grid.width * length_ratio)
            normal_text_line_length = max(normal_text_line_height, normal_text_line_length)

            pad_max = grid.width - normal_text_line_length
            pad = rng.integers(0, pad_max + 1)
            left = grid.left + pad
            right = left + normal_text_line_length - 1
            assert right <= grid.right

            text_line_idx = len(layout_text_lines)
            layout_text_lines.append(
                LayoutTextLine(
                    grid_idx=grid_idx,
                    text_line_idx=text_line_idx,
                    text_line_height=normal_text_line_height,
                    box=Box(up=up, down=down, left=left, right=right),
                    glyph_sequence=FontEngineRunConfigGlyphSequence.HORI_DEFAULT,
                )
            )

            prev_text_line_height = normal_text_line_height
            normal_text_line_heights_acc_areas[normal_text_line_height_idx] \
                += normal_text_line_length * normal_text_line_height
            up = down + 1

        return layout_text_lines

    def fill_large_text_line_to_grid(
        self,
        large_text_line_gird: Box,
        rng: RandomGenerator,
    ):
        length_ratio = rng.uniform(
            self.config.large_text_line_length_ratio_min,
            self.config.large_text_line_length_ratio_max,
        )
        large_text_line_length = round(large_text_line_gird.width * length_ratio)
        large_text_line_length = max(large_text_line_gird.height, large_text_line_length)

        pad_max = large_text_line_gird.width - large_text_line_length
        pad = rng.integers(0, pad_max + 1)
        left = large_text_line_gird.left + pad
        right = left + large_text_line_length - 1
        assert right <= large_text_line_gird.right

        return LayoutTextLine(
            grid_idx=-1,
            text_line_idx=0,
            text_line_height=large_text_line_gird.height,
            box=attrs.evolve(large_text_line_gird, left=left, right=right),
            glyph_sequence=FontEngineRunConfigGlyphSequence.HORI_DEFAULT,
        )

    def get_reference_height(self, height: int, width: int):
        area = height * width
        reference_height = math.ceil(math.sqrt(area / self.config.reference_aspect_ratio))
        return reference_height

    def sample_layout_text_lines(self, height: int, width: int, rng: RandomGenerator):
        reference_height = self.get_reference_height(height=height, width=width)

        normal_text_line_heights = self.sample_normal_text_line_heights(reference_height, rng)
        (vert_begins, vert_ends), (hori_begins, hori_ends) = self.sample_grid_points(
            height=height,
            width=width,
            normal_text_line_heights_max=normal_text_line_heights[-1],
            rng=rng,
        )

        large_text_line_height = self.sample_large_text_line_height(reference_height, rng)
        large_text_line_gird: Optional[Box] = None
        if large_text_line_height is not None:
            large_text_line_gird, vert_trim_idx = self.trim_grid_points_for_large_text_line(
                large_text_line_height=large_text_line_height,
                vert_begins=vert_begins,
                vert_ends=vert_ends,
                hori_begins_min=hori_begins[0],
                hori_ends_max=hori_ends[-1],
            )
            if large_text_line_gird is not None:
                vert_begins = vert_begins[vert_trim_idx:]
                vert_ends = vert_ends[vert_trim_idx:]

        grids = self.sample_grids(
            vert_begins=vert_begins,
            vert_ends=vert_ends,
            hori_begins=hori_begins,
            hori_ends=hori_ends,
            rng=rng,
        )
        normal_text_line_heights_expected_probs = normalize_to_probs([
            1 / normal_text_line_height for normal_text_line_height in normal_text_line_heights
        ])
        normal_text_line_heights_acc_areas = [0] * len(normal_text_line_heights)
        layout_text_lines: List[LayoutTextLine] = []
        for grid_idx, grid in enumerate(grids):
            layout_text_lines.extend(
                self.fill_normal_text_lines_to_grid(
                    normal_text_line_heights=normal_text_line_heights,
                    normal_text_line_heights_expected_probs=normal_text_line_heights_expected_probs,
                    normal_text_line_heights_acc_areas=normal_text_line_heights_acc_areas,
                    grid_idx=grid_idx,
                    grid=grid,
                    rng=rng,
                )
            )

        if large_text_line_gird:
            layout_text_lines.append(self.fill_large_text_line_to_grid(large_text_line_gird, rng))

        # Must place text line.
        assert layout_text_lines

        return (
            layout_text_lines,
            large_text_line_gird,
            grids,
        )

    def sample_layout_images(self, height: int, width: int, rng: RandomGenerator):
        # Image could be overlapped with text lines.
        layout_images: List[LayoutImage] = []

        num_layout_images = rng.integers(
            self.config.num_images_min,
            self.config.num_images_max + 1,
        )
        for _ in range(num_layout_images):
            # NOTE: It's ok to have overlapping images.
            image_height_ratio = rng.uniform(
                self.config.image_height_ratio_min,
                self.config.image_height_ratio_max,
            )
            image_height = round(height * image_height_ratio)

            image_width_ratio = rng.uniform(
                self.config.image_width_ratio_min,
                self.config.image_width_ratio_max,
            )
            image_width = round(width * image_width_ratio)

            up = rng.integers(0, height - image_height + 1)
            down = up + image_height - 1
            left = rng.integers(0, width - image_width + 1)
            right = left + image_width - 1
            layout_images.append(LayoutImage(box=Box(up=up, down=down, left=left, right=right)))

        return layout_images

    @classmethod
    def boxes_are_overlapped(cls, box0: Box, box1: Box):
        vert_overlapped = (box0.down >= box1.up and box1.down >= box0.up)
        hori_overlapped = (box0.right >= box1.left and box1.right >= box0.left)
        return vert_overlapped and hori_overlapped

    def sample_layout_barcode_qrs(
        self,
        height: int,
        width: int,
        layout_text_lines: Sequence[LayoutTextLine],
        rng: RandomGenerator,
    ):
        reference_height = self.get_reference_height(height=height, width=width)

        layout_barcode_qrs: List[LayoutBarcodeQr] = []

        num_layout_barcode_qrs = rng.integers(
            self.config.num_barcode_qrs_min,
            self.config.num_barcode_qrs_max + 1,
        )
        num_retries = 3
        while num_layout_barcode_qrs > 0 and num_retries > 0:
            barcode_qr_length_ratio = rng.uniform(
                self.config.barcode_qr_length_ratio_min,
                self.config.barcode_qr_length_ratio_max,
            )
            barcode_qr_length = round(barcode_qr_length_ratio * reference_height)
            barcode_qr_length = min(height, width, barcode_qr_length)

            # Place QR code next to text line.
            anchor_layout_text_line_box = rng_choice(rng, layout_text_lines).box
            anchor_layout_text_line_box_center = anchor_layout_text_line_box.get_center_point()
            placement = rng_choice(rng, tuple(LayoutXcodePlacement))

            if placement in (LayoutXcodePlacement.NEXT_TO_DOWN, LayoutXcodePlacement.NEXT_TO_UP):
                if placement == LayoutXcodePlacement.NEXT_TO_DOWN:
                    up = anchor_layout_text_line_box.down + 1
                    down = up + barcode_qr_length - 1
                    if down >= height:
                        num_retries -= 1
                        continue
                else:
                    assert placement == LayoutXcodePlacement.NEXT_TO_UP
                    down = anchor_layout_text_line_box.up - 1
                    up = down + 1 - barcode_qr_length
                    if up < 0:
                        num_retries -= 1
                        continue

                left_min = max(
                    0,
                    anchor_layout_text_line_box_center.x - barcode_qr_length,
                )
                left_max = min(
                    width - barcode_qr_length,
                    anchor_layout_text_line_box_center.x,
                )
                if left_min > left_max:
                    num_retries -= 1
                    continue
                left = int(rng.integers(left_min, left_max + 1))
                right = left + barcode_qr_length - 1

            else:
                assert placement in (
                    LayoutXcodePlacement.NEXT_TO_RIGHT,
                    LayoutXcodePlacement.NEXT_TO_LEFT,
                )

                if placement == LayoutXcodePlacement.NEXT_TO_RIGHT:
                    left = anchor_layout_text_line_box.right + 1
                    right = left + barcode_qr_length - 1
                    if right >= width:
                        num_retries -= 1
                        continue
                else:
                    assert placement == LayoutXcodePlacement.NEXT_TO_LEFT
                    right = anchor_layout_text_line_box.left - 1
                    left = right + 1 - barcode_qr_length
                    if left < 0:
                        num_retries -= 1
                        continue

                up_min = max(
                    0,
                    anchor_layout_text_line_box_center.y - barcode_qr_length,
                )
                up_max = min(
                    height - barcode_qr_length,
                    anchor_layout_text_line_box_center.y,
                )
                if up_min > up_max:
                    num_retries -= 1
                    continue

                up = int(rng.integers(up_min, up_max + 1))
                down = up + barcode_qr_length - 1

            num_layout_barcode_qrs -= 1
            layout_barcode_qrs.append(
                LayoutBarcodeQr(box=Box(
                    up=up,
                    down=down,
                    left=left,
                    right=right,
                ))
            )

        return layout_barcode_qrs

    def sample_layout_barcode_code39s(
        self,
        height: int,
        width: int,
        layout_text_lines: Sequence[LayoutTextLine],
        rng: RandomGenerator,
    ):
        reference_height = self.get_reference_height(height=height, width=width)

        layout_barcode_code39s: List[LayoutBarcodeCode39] = []

        num_layout_barcode_code39s = rng.integers(
            self.config.num_barcode_code39s_min,
            self.config.num_barcode_code39s_max + 1,
        )
        num_retries = 3
        while num_layout_barcode_code39s > 0 and num_retries > 0:
            barcode_code39_height_ratio = rng.uniform(
                self.config.barcode_code39_height_ratio_min,
                self.config.barcode_code39_height_ratio_max,
            )
            barcode_code39_height = round(barcode_code39_height_ratio * reference_height)
            barcode_code39_height = min(height, width, barcode_code39_height)

            barcode_code39_num_chars = int(
                rng.integers(
                    self.config.barcode_code39_num_chars_min,
                    self.config.barcode_code39_num_chars_max + 1,
                )
            )
            barcode_code39_width = round(
                barcode_code39_height * self.config.barcode_code39_aspect_ratio
                * barcode_code39_num_chars
            )

            # Place Bar code next to text line.
            anchor_layout_text_line_box = rng_choice(rng, layout_text_lines).box
            anchor_layout_text_line_box_center = anchor_layout_text_line_box.get_center_point()
            placement = rng_choice(rng, tuple(LayoutXcodePlacement))

            if placement in (LayoutXcodePlacement.NEXT_TO_DOWN, LayoutXcodePlacement.NEXT_TO_UP):
                if placement == LayoutXcodePlacement.NEXT_TO_DOWN:
                    up = anchor_layout_text_line_box.down + 1
                    down = up + barcode_code39_height - 1
                    if down >= height:
                        num_retries -= 1
                        continue
                else:
                    assert placement == LayoutXcodePlacement.NEXT_TO_UP
                    down = anchor_layout_text_line_box.up - 1
                    up = down + 1 - barcode_code39_height
                    if up < 0:
                        num_retries -= 1
                        continue

                left_min = max(
                    0,
                    anchor_layout_text_line_box_center.x - barcode_code39_width,
                )
                left_max = min(
                    width - barcode_code39_width,
                    anchor_layout_text_line_box_center.x,
                )
                if left_min > left_max:
                    num_retries -= 1
                    continue
                left = int(rng.integers(left_min, left_max + 1))
                right = left + barcode_code39_width - 1

            else:
                assert placement in (
                    LayoutXcodePlacement.NEXT_TO_RIGHT,
                    LayoutXcodePlacement.NEXT_TO_LEFT,
                )

                if placement == LayoutXcodePlacement.NEXT_TO_RIGHT:
                    left = anchor_layout_text_line_box.right + 1
                    right = left + barcode_code39_width - 1
                    if right >= width:
                        num_retries -= 1
                        continue
                else:
                    assert placement == LayoutXcodePlacement.NEXT_TO_LEFT
                    right = anchor_layout_text_line_box.left - 1
                    left = right + 1 - barcode_code39_width
                    if left < 0:
                        num_retries -= 1
                        continue

                up_min = max(
                    0,
                    anchor_layout_text_line_box_center.y - barcode_code39_height,
                )
                up_max = min(
                    height - barcode_code39_height,
                    anchor_layout_text_line_box_center.y,
                )
                if up_min > up_max:
                    num_retries -= 1
                    continue

                up = int(rng.integers(up_min, up_max + 1))
                down = up + barcode_code39_height - 1

            num_layout_barcode_code39s -= 1
            layout_barcode_code39s.append(
                LayoutBarcodeCode39(box=Box(
                    up=up,
                    down=down,
                    left=left,
                    right=right,
                ))
            )

        return layout_barcode_code39s

    def sample_layout_barcode_qrs_and_layout_barcode_code39s(
        self,
        height: int,
        width: int,
        layout_text_lines: Sequence[LayoutTextLine],
        rng: RandomGenerator,
    ):
        layout_barcode_qrs = self.sample_layout_barcode_qrs(
            height=height,
            width=width,
            layout_text_lines=layout_text_lines,
            rng=rng,
        )

        layout_barcode_code39s = self.sample_layout_barcode_code39s(
            height=height,
            width=width,
            layout_text_lines=layout_text_lines,
            rng=rng,
        )

        if layout_barcode_qrs or layout_barcode_code39s:
            # Barcode could not be overlapped with text lines.
            # Hence need to remove the overlapped text lines.
            box_overlapping_validator = BoxOverlappingValidator(
                itertools.chain(
                    (layout_barcode_qr.box for layout_barcode_qr in layout_barcode_qrs),
                    (layout_barcode_code39.box for layout_barcode_code39 in layout_barcode_code39s),
                )
            )

            keep_layout_text_lines: List[LayoutTextLine] = []
            for layout_text_line in layout_text_lines:
                if not box_overlapping_validator.is_overlapped(layout_text_line.box):
                    keep_layout_text_lines.append(layout_text_line)
            layout_text_lines = keep_layout_text_lines

        return layout_barcode_qrs, layout_barcode_code39s, layout_text_lines

    @classmethod
    def get_text_line_area(cls, layout_text_lines: Sequence[LayoutTextLine]):
        # Sample within the text line area.
        text_line_up = min(layout_text_line.box.up for layout_text_line in layout_text_lines)
        text_line_down = max(layout_text_line.box.down for layout_text_line in layout_text_lines)
        text_line_left = min(layout_text_line.box.left for layout_text_line in layout_text_lines)
        text_line_right = max(layout_text_line.box.right for layout_text_line in layout_text_lines)
        return (
            text_line_up,
            text_line_down,
            text_line_left,
            text_line_right,
        )

    def sample_layout_non_text_symbols(
        self,
        height: int,
        width: int,
        layout_text_lines: Sequence[LayoutTextLine],
        rng: RandomGenerator,
    ):
        reference_height = self.get_reference_height(height=height, width=width)

        text_line_up = 0
        text_line_down = height - 1
        text_line_left = 0
        text_line_right = width - 1

        layout_non_text_symbols: List[LayoutNonTextSymbol] = []

        num_non_text_symbols = int(
            rng.integers(
                self.config.num_non_text_symbols_min,
                self.config.num_non_text_symbols_max + 1,
            )
        )
        for _ in range(num_non_text_symbols):
            non_text_symbol_height_ratio = rng.uniform(
                self.config.non_text_symbol_height_ratio_min,
                self.config.non_text_symbol_height_ratio_max,
            )
            non_text_symbol_height = round(non_text_symbol_height_ratio * reference_height)

            non_text_symbol_aspect_ratio = rng.uniform(
                self.config.non_text_symbol_aspect_ratio_min,
                self.config.non_text_symbol_aspect_ratio_max,
            )
            non_text_symbol_width = round(non_text_symbol_aspect_ratio * non_text_symbol_height)

            box = None
            overlapped = True
            for _ in range(self.config.num_retries_to_get_non_overlapped_non_text_symbol):
                up_max = text_line_down + 1 - non_text_symbol_height
                up = int(rng.integers(text_line_up, up_max + 1))
                down = up + non_text_symbol_height - 1
                assert up < down

                left_max = text_line_right + 1 - non_text_symbol_width
                left = int(rng.integers(text_line_left, left_max + 1))
                right = left + non_text_symbol_width - 1
                assert left < right

                box = Box(up=up, down=down, left=left, right=right)

                cur_overlapped = False
                for layout_text_line in layout_text_lines:
                    if self.boxes_are_overlapped(box, layout_text_line.box):
                        cur_overlapped = True
                        break

                if not cur_overlapped:
                    overlapped = False
                    break

            assert box

            if not overlapped:
                alpha = float(
                    rng.uniform(
                        self.config.non_text_symbol_non_overlapped_alpha_min,
                        self.config.non_text_symbol_non_overlapped_alpha_max,
                    )
                )
            else:
                alpha = float(
                    rng.uniform(
                        self.config.non_text_symbol_overlapped_alpha_min,
                        self.config.non_text_symbol_overlapped_alpha_max,
                    )
                )

            layout_non_text_symbols.append(LayoutNonTextSymbol(
                box=box,
                alpha=alpha,
            ))

        return layout_non_text_symbols

    def sample_layout_seal_impressions(
        self,
        height: int,
        width: int,
        layout_text_lines: Sequence[LayoutTextLine],
        rng: RandomGenerator,
    ):
        reference_height = self.get_reference_height(height=height, width=width)

        (
            text_line_up,
            text_line_down,
            text_line_left,
            text_line_right,
        ) = self.get_text_line_area(layout_text_lines)

        # Place seal impressions.
        layout_seal_impressions: List[LayoutSealImpression] = []

        num_seal_impressions = int(
            rng.integers(
                self.config.num_seal_impressions_min,
                self.config.num_seal_impressions_max + 1,
            )
        )
        for _ in range(num_seal_impressions):
            # Sample height.
            seal_impression_height_ratio = float(
                rng.uniform(
                    self.config.seal_impression_height_ratio_min,
                    self.config.seal_impression_height_ratio_max,
                )
            )
            seal_impression_height = round(seal_impression_height_ratio * reference_height)
            seal_impression_height = min(text_line_down + 1 - text_line_up, seal_impression_height)

            # Make sure even.
            if seal_impression_height % 2 != 0:
                seal_impression_height -= 1

            # Sample width.
            shape_mode = rng_choice(
                rng,
                self.seal_impression_ellipse_shape_modes,
                probs=self.seal_impression_ellipse_shape_modes_probs,
            )
            if shape_mode == SealImpressionEllipseShapeMode.CIRCLE:
                seal_impression_width = seal_impression_height

            elif shape_mode == SealImpressionEllipseShapeMode.GENERAL_ELLIPSE:
                aspect_ratio = float(
                    rng.uniform(
                        self.config.seal_impression_general_ellipse_aspect_ratio_min,
                        self.config.seal_impression_general_ellipse_aspect_ratio_max,
                    )
                )
                seal_impression_width = round(aspect_ratio * seal_impression_height)

            else:
                raise NotImplementedError()

            seal_impression_width = min(text_line_right + 1 - text_line_left, seal_impression_width)

            # Make sure even.
            if seal_impression_width % 2 != 0:
                seal_impression_width -= 1

            seal_impression_up_max = text_line_down + 1 - seal_impression_height
            seal_impression_up = int(rng.integers(
                text_line_up,
                seal_impression_up_max + 1,
            ))
            seal_impression_down = seal_impression_up + seal_impression_height - 1

            seal_impression_left_max = text_line_right + 1 - seal_impression_width
            seal_impression_left = int(rng.integers(
                text_line_left,
                seal_impression_left_max + 1,
            ))
            seal_impression_right = seal_impression_left + seal_impression_width - 1

            angle = int(
                rng.integers(
                    self.config.seal_impression_angle_min,
                    self.config.seal_impression_angle_max + 1,
                )
            )
            angle = angle % 360

            layout_seal_impressions.append(
                LayoutSealImpression(
                    box=Box(
                        up=seal_impression_up,
                        down=seal_impression_down,
                        left=seal_impression_left,
                        right=seal_impression_right,
                    ),
                    angle=angle,
                )
            )

        return layout_seal_impressions

    def generate_disconnected_text_regions(
        self,
        layout_text_lines: Sequence[LayoutTextLine],
    ):
        grid_idx_to_layout_text_lines: DefaultDict[int, List[LayoutTextLine]] = defaultdict(list)
        for layout_text_line in layout_text_lines:
            grid_idx_to_layout_text_lines[layout_text_line.grid_idx].append(layout_text_line)

        disconnected_text_regions: List[DisconnectedTextRegion] = []

        for _, layout_text_lines in sorted(
            grid_idx_to_layout_text_lines.items(),
            key=lambda p: p[0],
        ):
            layout_text_lines = sorted(layout_text_lines, key=lambda ltl: ltl.text_line_idx)

            begin = 0
            while begin < len(layout_text_lines):
                text_line_height_min = layout_text_lines[begin].text_line_height
                text_line_height_max = text_line_height_min

                # Find [begin, end) interval satisfying the condition.
                end = begin + 1
                while end < len(layout_text_lines):
                    text_line_height = layout_text_lines[end].text_line_height
                    text_line_height_min = min(text_line_height_min, text_line_height)
                    text_line_height_max = max(text_line_height_max, text_line_height)
                    if text_line_height_max / text_line_height_min \
                            > self.config.disconnected_text_region_polygons_height_ratio_max:
                        break
                    else:
                        end += 1

                # To polygon.
                # NOTE: Simply using a bounding box is enough.
                # This method is common to all glyph sequences.
                cur_layout_text_lines = layout_text_lines[begin:end]
                bounding_box = Box(
                    up=min(ltl.box.up for ltl in cur_layout_text_lines),
                    down=max(ltl.box.down for ltl in cur_layout_text_lines),
                    left=min(ltl.box.left for ltl in cur_layout_text_lines),
                    right=max(ltl.box.right for ltl in cur_layout_text_lines),
                )
                step = min(
                    itertools.chain.from_iterable(ltl.box.shape for ltl in cur_layout_text_lines)
                )
                disconnected_text_regions.append(
                    DisconnectedTextRegion(polygon=bounding_box.to_polygon(step=step))
                )

                # Move to next.
                begin = end

        return disconnected_text_regions

    def generate_non_text_regions(
        self,
        height: int,
        width: int,
        layout_text_lines: Sequence[LayoutTextLine],
        rng: RandomGenerator,
    ):
        box_overlapping_validator = BoxOverlappingValidator(
            layout_text_line.box for layout_text_line in layout_text_lines
        )
        directions = [
            LayoutNonTextLineDirection.UP,
            LayoutNonTextLineDirection.DOWN,
            LayoutNonTextLineDirection.LEFT,
            LayoutNonTextLineDirection.RIGHT,
        ]

        lntl_boxes: List[Box] = []
        for layout_text_line in layout_text_lines:
            ltl_box = layout_text_line.box

            for direction_idx in rng.permutation(len(directions)):
                direction = directions[direction_idx]

                if direction == LayoutNonTextLineDirection.UP:
                    lntl_box = Box(
                        up=ltl_box.up - ltl_box.height,
                        down=ltl_box.up - 1,
                        left=ltl_box.left,
                        right=ltl_box.right,
                    )

                elif direction == LayoutNonTextLineDirection.DOWN:
                    lntl_box = Box(
                        up=ltl_box.down + 1,
                        down=ltl_box.down + ltl_box.height,
                        left=ltl_box.left,
                        right=ltl_box.right,
                    )

                elif direction == LayoutNonTextLineDirection.LEFT:
                    lntl_box = Box(
                        up=ltl_box.up,
                        down=ltl_box.down,
                        left=ltl_box.left - ltl_box.width,
                        right=ltl_box.left - 1,
                    )

                elif direction == LayoutNonTextLineDirection.RIGHT:
                    lntl_box = Box(
                        up=ltl_box.up,
                        down=ltl_box.down,
                        left=ltl_box.right + 1,
                        right=ltl_box.right + ltl_box.width,
                    )

                else:
                    raise NotImplementedError()

                # Ignore invalid box.
                if not lntl_box.valid:
                    continue
                if lntl_box.down >= height or lntl_box.right >= width:
                    continue

                assert ltl_box.shape == lntl_box.shape

                # Ignore box that is overlapped with any text lines.
                if box_overlapping_validator.is_overlapped(lntl_box):
                    continue

                # Keep only the first valid direction.
                lntl_boxes.append(lntl_box)
                break

        step = max(
            1,
            min(itertools.chain.from_iterable(lntl_box.shape for lntl_box in lntl_boxes)),
        )
        non_text_regions = [
            NonTextRegion(polygon=lntl_box.to_polygon(step=step)) for lntl_box in lntl_boxes
        ]
        return non_text_regions

    def run(self, input: PageLayoutStepInput, rng: RandomGenerator):
        page_shape_step_output = input.page_shape_step_output
        height = page_shape_step_output.height
        width = page_shape_step_output.width

        # Text lines.
        (
            layout_text_lines,
            large_text_line_gird,
            grids,
        ) = self.sample_layout_text_lines(height=height, width=width, rng=rng)

        # Images.
        layout_images = self.sample_layout_images(height=height, width=width, rng=rng)

        # QR codes & Bar codes.
        # NOTE: Some layout_text_lines could be dropped.
        (
            layout_barcode_qrs,
            layout_barcode_code39s,
            layout_text_lines,
        ) = self.sample_layout_barcode_qrs_and_layout_barcode_code39s(
            height=height,
            width=width,
            layout_text_lines=layout_text_lines,
            rng=rng,
        )

        # Non-text symbols.
        layout_non_text_symbols = self.sample_layout_non_text_symbols(
            height=height,
            width=width,
            layout_text_lines=layout_text_lines,
            rng=rng,
        )

        # Seal impressions.
        layout_seal_impressions = self.sample_layout_seal_impressions(
            height=height,
            width=width,
            layout_text_lines=layout_text_lines,
            rng=rng,
        )

        # For char-level polygon regression.
        disconnected_text_regions = self.generate_disconnected_text_regions(
            layout_text_lines=layout_text_lines,
        )

        # For sampling negative text region area.
        non_text_regions = self.generate_non_text_regions(
            height=height,
            width=width,
            layout_text_lines=layout_text_lines,
            rng=rng,
        )

        return PageLayoutStepOutput(
            page_layout=PageLayout(
                height=height,
                width=width,
                layout_text_lines=layout_text_lines,
                layout_non_text_symbols=layout_non_text_symbols,
                layout_seal_impressions=layout_seal_impressions,
                layout_images=layout_images,
                layout_barcode_qrs=layout_barcode_qrs,
                layout_barcode_code39s=layout_barcode_code39s,
                disconnected_text_regions=disconnected_text_regions,
                non_text_regions=non_text_regions,
            ),
            debug_large_text_line_gird=large_text_line_gird,
            debug_grids=grids,
        )


page_layout_step_factory = PipelineStepFactory(PageLayoutStep)
