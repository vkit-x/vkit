from typing import Optional, Sequence, List
import math
import heapq

import attrs
from numpy.random import RandomState

from vkit.utility import rnd_choice, normalize_to_probs
from vkit.element import Box
from vkit.engine.font.type import FontEngineRunConfigGlyphSequence
from .page_shape import PageShapeStep
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)


@attrs.define
class PageLayoutStepConfig:
    # Text line heights.
    reference_aspect_ratio: float = 1 / 1.4142

    # Large text line.
    prob_add_large_text_line: float = 0.5
    large_text_line_height_ratio_min: float = 0.1
    large_text_line_height_ratio_max: float = 0.2
    large_text_line_length_ratio_min: float = 0.5
    large_text_line_length_ratio_max: float = 1.0

    # Normal text line.
    num_normal_text_line_heights_min: int = 2
    num_normal_text_line_heights_max: int = 4
    normal_text_line_height_ratio_min: float = 0.01
    normal_text_line_height_ratio_max: float = 0.08
    force_add_normal_text_line_height_ratio_min: bool = True

    prob_normal_text_line_diff_heights_gap: float = 0.5
    prob_normal_text_line_gap: float = 0.5
    normal_text_line_gap_ratio_min: float = 0.0
    normal_text_line_gap_ratio_max: float = 0.5
    normal_text_line_length_ratio_min: float = 0.5
    normal_text_line_length_ratio_max: float = 1.0

    # Grid points.
    grid_pad_ratio_min: float = 0.01
    grid_pad_ratio_max: float = 0.05
    grid_step_ratio_min: float = 1.0
    grid_step_ratio_max: float = 1.1
    grid_vert_gap_ratio_min: float = 0.0
    grid_vert_gap_ratio_max: float = 0.5
    grid_hori_gap_ratio_min: float = 1.0
    grid_hori_gap_ratio_max: float = 1.15

    # Image.
    num_images_min: int = 0
    num_images_max: int = 3
    image_height_ratio_min: float = 0.1
    image_height_ratio_max: float = 0.35
    image_width_ratio_min: float = 0.1
    image_width_ratio_max: float = 0.35


@attrs.define
class LayoutImage:
    box: Box


@attrs.define
class LayoutTextLine:
    box: Box
    glyph_sequence: FontEngineRunConfigGlyphSequence


@attrs.define
class PageLayout:
    height: int
    width: int
    layout_images: Sequence[LayoutImage]
    layout_text_lines: Sequence[LayoutTextLine]


@attrs.define
class PageLayoutStepOutput:
    page_layout: PageLayout
    debug_large_text_line_gird: Optional[Box]
    debug_normal_grids: Sequence[Box]


@attrs.define(order=True)
class PrioritizedSegment:
    vert_begin_idx: int = attrs.field(order=True)
    hori_begin_idx: int = attrs.field(order=False)
    hori_end_idx: int = attrs.field(order=False)


class PageLayoutStep(
    PipelineStep[
        PageLayoutStepConfig,
        PageLayoutStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageLayoutStepConfig):
        super().__init__(config)

    def sample_large_text_line_height(self, reference_height: int, rnd: RandomState):
        if rnd.random() < self.config.prob_add_large_text_line:
            large_text_line_height_ratio = rnd.uniform(
                self.config.large_text_line_height_ratio_min,
                self.config.large_text_line_height_ratio_max,
            )
            return round(large_text_line_height_ratio * reference_height)

        else:
            return None

    def sample_normal_text_line_heights(self, reference_height: int, rnd: RandomState):
        normal_text_line_heights: List[int] = []

        if self.config.force_add_normal_text_line_height_ratio_min:
            normal_text_line_heights.append(
                round(self.config.normal_text_line_height_ratio_min * reference_height)
            )

        num_normal_text_line_heights = rnd.randint(
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
            ratio = rnd.uniform(ratio_min, ratio_max)
            normal_text_line_heights.append(round(ratio * reference_height))

        assert normal_text_line_heights
        return sorted(normal_text_line_heights)

    @staticmethod
    def generate_grid_points(
        grid_pad_ratio: float,
        grid_step: int,
        grid_gap: int,
        grid_gap_min: Optional[int],
        length: int,
        rnd: RandomState,
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
                cur_gap = rnd.randint(grid_gap_min, grid_gap + 1)

            begin = end + cur_gap
            end = begin + grid_step - 1

        return begins, ends

    def sample_grid_points(
        self,
        height: int,
        width: int,
        normal_text_line_heights_max: int,
        rnd: RandomState,
    ):
        grid_pad_ratio = rnd.uniform(
            self.config.grid_pad_ratio_min,
            self.config.grid_pad_ratio_max,
        )

        grid_step_ratio = rnd.uniform(
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
            rnd=rnd,
        )

        grid_hori_gap_ratio = rnd.uniform(
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
            rnd=rnd,
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

    def sample_normal_grids(
        self,
        vert_begins: Sequence[int],
        vert_ends: Sequence[int],
        hori_begins: Sequence[int],
        hori_ends: Sequence[int],
        rnd: RandomState,
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
                cur_vert_end_idx = rnd.randint(vert_begin_idx, num_vert_ends)

                # Try to sample segment with length >= 2.
                if hori_end_idx + 1 - hori_begin_idx <= 3:
                    cur_hori_end_idx = hori_end_idx
                else:
                    cur_hori_end_idx = rnd.randint(hori_begin_idx + 1, hori_end_idx + 1)

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

    @staticmethod
    def calculate_normal_text_line_heights_probs(
        normal_text_line_heights_expected_probs: Sequence[float],
        normal_text_line_heights_acc_lengths: List[int],
    ):
        if sum(normal_text_line_heights_acc_lengths) == 0:
            normal_text_line_heights_cur_probs = [0.0] * len(normal_text_line_heights_acc_lengths)
        else:
            normal_text_line_heights_cur_probs = normalize_to_probs(
                normal_text_line_heights_acc_lengths
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
        normal_text_line_heights_acc_lengths: List[int],
        normal_grid: Box,
        rnd: RandomState,
    ):
        normal_text_line_heights_indices = list(range(len(normal_text_line_heights)))
        normal_text_line_heights_max = normal_text_line_heights[-1]

        layout_text_lines: List[LayoutTextLine] = []
        up = normal_grid.up
        prev_text_line_height: Optional[int] = None

        while up + normal_text_line_heights_max - 1 <= normal_grid.down:
            normal_text_line_heights_probs = self.calculate_normal_text_line_heights_probs(
                normal_text_line_heights_expected_probs=normal_text_line_heights_expected_probs,
                normal_text_line_heights_acc_lengths=normal_text_line_heights_acc_lengths,
            )
            normal_text_line_height_idx = rnd_choice(
                rnd=rnd,
                items=normal_text_line_heights_indices,
                probs=normal_text_line_heights_probs,
            )
            normal_text_line_height = normal_text_line_heights[normal_text_line_height_idx]

            add_gap = False
            if prev_text_line_height:
                if prev_text_line_height != normal_text_line_height:
                    add_gap = (rnd.random() < self.config.prob_normal_text_line_diff_heights_gap)
                else:
                    add_gap = (rnd.random() < self.config.prob_normal_text_line_gap)
            if add_gap:
                gap_ratio = rnd.uniform(
                    self.config.normal_text_line_gap_ratio_min,
                    self.config.normal_text_line_gap_ratio_max,
                )
                gap = round(gap_ratio * normal_text_line_height)
                gap = min(normal_grid.down - (up + normal_text_line_height - 1), gap)
                up += gap
            down = up + normal_text_line_height - 1
            assert down <= normal_grid.down

            length_ratio = rnd.uniform(
                self.config.normal_text_line_length_ratio_min,
                self.config.normal_text_line_length_ratio_max,
            )
            normal_text_line_length = round(normal_grid.width * length_ratio)
            normal_text_line_length = max(normal_text_line_height, normal_text_line_length)

            pad_max = normal_grid.width - normal_text_line_length
            pad = rnd.randint(0, pad_max + 1)
            left = normal_grid.left + pad
            right = left + normal_text_line_length - 1
            assert right <= normal_grid.right

            layout_text_lines.append(
                LayoutTextLine(
                    box=Box(up=up, down=down, left=left, right=right),
                    glyph_sequence=FontEngineRunConfigGlyphSequence.HORI_DEFAULT,
                )
            )

            prev_text_line_height = normal_text_line_height
            normal_text_line_heights_acc_lengths[normal_text_line_height_idx] \
                += normal_text_line_length
            up = down + 1

        return layout_text_lines

    def fill_large_text_line_to_grid(
        self,
        large_text_line_gird: Box,
        rnd: RandomState,
    ):
        length_ratio = rnd.uniform(
            self.config.large_text_line_length_ratio_min,
            self.config.large_text_line_length_ratio_max,
        )
        large_text_line_length = round(large_text_line_gird.width * length_ratio)
        large_text_line_length = max(large_text_line_gird.height, large_text_line_length)

        pad_max = large_text_line_gird.width - large_text_line_length
        pad = rnd.randint(0, pad_max + 1)
        left = large_text_line_gird.left + pad
        right = left + large_text_line_length - 1
        assert right <= large_text_line_gird.right

        return LayoutTextLine(
            box=attrs.evolve(large_text_line_gird, left=left, right=right),
            glyph_sequence=FontEngineRunConfigGlyphSequence.HORI_DEFAULT,
        )

    def sample_layout_images(self, height: int, width: int, rnd: RandomState):
        layout_images: List[LayoutImage] = []

        num_layout_images = rnd.randint(
            self.config.num_images_min,
            self.config.num_images_max + 1,
        )
        for _ in range(num_layout_images):
            # NOTE: It's ok to have overlapping images.
            image_height_ratio = rnd.uniform(
                self.config.image_height_ratio_min,
                self.config.image_height_ratio_max,
            )
            image_height = round(height * image_height_ratio)

            image_width_ratio = rnd.uniform(
                self.config.image_width_ratio_min,
                self.config.image_width_ratio_max,
            )
            image_width = round(width * image_width_ratio)

            up = rnd.randint(0, height - image_height + 1)
            down = up + image_height - 1
            left = rnd.randint(0, width - image_width + 1)
            right = left + image_width - 1
            layout_images.append(LayoutImage(box=Box(up=up, down=down, left=left, right=right)))

        return layout_images

    def run(self, state: PipelineState, rnd: RandomState):
        page_shape_step_output = self.get_output(state, PageShapeStep)
        height = page_shape_step_output.height
        width = page_shape_step_output.width

        area = height * width
        reference_height = round(math.sqrt(area / self.config.reference_aspect_ratio))

        normal_text_line_heights = self.sample_normal_text_line_heights(reference_height, rnd)
        (vert_begins, vert_ends), (hori_begins, hori_ends) = self.sample_grid_points(
            height=height,
            width=width,
            normal_text_line_heights_max=normal_text_line_heights[-1],
            rnd=rnd,
        )

        large_text_line_height = self.sample_large_text_line_height(reference_height, rnd)
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

        normal_grids = self.sample_normal_grids(
            vert_begins=vert_begins,
            vert_ends=vert_ends,
            hori_begins=hori_begins,
            hori_ends=hori_ends,
            rnd=rnd,
        )
        normal_text_line_heights_expected_probs = normalize_to_probs([
            1 / normal_text_line_height for normal_text_line_height in normal_text_line_heights
        ])
        normal_text_line_heights_acc_lengths = [0] * len(normal_text_line_heights)
        layout_text_lines: List[LayoutTextLine] = []
        for normal_grid in normal_grids:
            layout_text_lines.extend(
                self.fill_normal_text_lines_to_grid(
                    normal_text_line_heights=normal_text_line_heights,
                    normal_text_line_heights_expected_probs=normal_text_line_heights_expected_probs,
                    normal_text_line_heights_acc_lengths=normal_text_line_heights_acc_lengths,
                    normal_grid=normal_grid,
                    rnd=rnd,
                )
            )

        if large_text_line_gird:
            layout_text_lines.append(self.fill_large_text_line_to_grid(large_text_line_gird, rnd))

        layout_images = self.sample_layout_images(height=height, width=width, rnd=rnd)

        return PageLayoutStepOutput(
            page_layout=PageLayout(
                height=height,
                width=width,
                layout_images=layout_images,
                layout_text_lines=layout_text_lines,
            ),
            debug_large_text_line_gird=large_text_line_gird,
            debug_normal_grids=normal_grids,
        )


page_layout_step_factory = PipelineStepFactory(PageLayoutStep)
