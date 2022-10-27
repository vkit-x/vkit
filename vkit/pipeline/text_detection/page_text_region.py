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
from typing import List, Optional, Dict, DefaultDict, Sequence, Tuple, Set
from collections import defaultdict
import itertools
import math
import statistics
import logging
import warnings

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
from shapely.errors import ShapelyDeprecationWarning
from shapely.strtree import STRtree
from shapely.geometry import Polygon as ShapelyPolygon
from rectpack import newPacker as RectPacker

from vkit.utility import rng_choice, rng_choice_with_size
from vkit.element import Box, Polygon, Mask, Image, ElementSetOperationMode
from vkit.mechanism.distortion import rotate
from ..interface import PipelineStep, PipelineStepFactory
from .page_distortion import PageDistortionStepOutput
from .page_resizing import PageResizingStepOutput

logger = logging.getLogger(__name__)

# Shapely version has been explicitly locked under 2.0, hence ignore this warning.
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)


@attrs.define
class PageTextRegionStepConfig:
    text_region_flattener_typical_long_side_ratio_min: float = 3.0
    text_region_flattener_text_region_polygon_dilate_ratio_min: float = 0.85
    text_region_flattener_text_region_polygon_dilate_ratio_max: float = 1.0
    text_region_resize_char_height_median_min: int = 30
    text_region_resize_char_height_median_max: int = 45
    text_region_typical_post_rotate_prob: float = 0.2
    text_region_untypical_post_rotate_prob: float = 0.2
    negative_text_region_ratio: float = 0.1
    negative_text_region_post_rotate_prob: float = 0.2
    stack_flattened_text_regions_pad: int = 2
    enable_post_rotate: bool = False
    post_rotate_angle_min: int = -10
    post_rotate_angle_max: int = 10
    enable_debug: bool = False


@attrs.define
class PageTextRegionStepInput:
    page_distortion_step_output: PageDistortionStepOutput
    page_resizing_step_output: PageResizingStepOutput


@attrs.define
class PageTextRegionInfo:
    precise_text_region_polygon: Polygon
    char_polygons: Sequence[Polygon]


@attrs.define
class FlattenedTextRegion:
    is_typical: bool
    text_region_polygon: Polygon
    text_region_image: Image
    bounding_extended_text_region_mask: Mask
    flattening_rotate_angle: int
    shape_before_trim: Tuple[int, int]
    rotated_trimmed_box: Box
    shape_before_resize: Tuple[int, int]
    post_rotate_angle: int
    flattened_image: Image
    flattened_mask: Mask
    flattened_char_polygons: Optional[Sequence[Polygon]]

    @property
    def shape(self):
        return self.flattened_image.shape

    @property
    def height(self):
        return self.flattened_image.height

    @property
    def width(self):
        return self.flattened_image.width

    @property
    def area(self):
        return self.flattened_image.area

    def get_char_height_meidan(self):
        assert self.flattened_char_polygons
        return statistics.median(
            char_polygon.get_rectangular_height() for char_polygon in self.flattened_char_polygons
        )

    def to_resized_flattened_text_region(
        self,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        resized_flattened_image = self.flattened_image.to_resized_image(
            resized_height=resized_height,
            resized_width=resized_width,
        )

        resized_flattened_mask = self.flattened_mask.to_resized_mask(
            resized_height=resized_height,
            resized_width=resized_width,
        )

        resized_flattened_char_polygons = None
        if self.flattened_char_polygons is not None:
            resized_flattened_char_polygons = [
                flattened_char_polygon.to_conducted_resized_polygon(
                    self.shape,
                    resized_height=resized_height,
                    resized_width=resized_width,
                ) for flattened_char_polygon in self.flattened_char_polygons
            ]

        return attrs.evolve(
            self,
            flattened_image=resized_flattened_image,
            flattened_mask=resized_flattened_mask,
            flattened_char_polygons=resized_flattened_char_polygons,
        )

    def to_post_rotated_flattened_text_region(
        self,
        post_rotate_angle: int,
    ):
        assert self.post_rotate_angle == 0

        # NOTE: No need to trim.
        rotated_result = rotate.distort(
            {'angle': post_rotate_angle},
            image=self.flattened_image,
            mask=self.flattened_mask,
            polygons=self.flattened_char_polygons,
        )
        rotated_flattened_image = rotated_result.image
        assert rotated_flattened_image
        rotated_flattened_mask = rotated_result.mask
        assert rotated_flattened_mask
        rotated_flattened_char_polygons = rotated_result.polygons

        return attrs.evolve(
            self,
            post_rotate_angle=post_rotate_angle,
            flattened_image=rotated_flattened_image,
            flattened_mask=rotated_flattened_mask,
            flattened_char_polygons=rotated_flattened_char_polygons,
        )


@attrs.define
class PageTextRegionStepDebug:
    page_image: Image = attrs.field(default=None)
    precise_text_region_candidate_polygons: Sequence[Polygon] = attrs.field(default=None)
    page_text_region_infos: Sequence[PageTextRegionInfo] = attrs.field(default=None)
    flattened_text_regions: Sequence[FlattenedTextRegion] = attrs.field(default=None)


@attrs.define
class PageTextRegionStepOutput:
    page_image: Image
    page_char_polygons: Sequence[Polygon]
    shape_before_rotate: Tuple[int, int]
    rotate_angle: int
    debug: Optional[PageTextRegionStepDebug]


def calculate_boxed_masks_intersected_ratio(
    anchor_mask: Mask,
    candidate_mask: Mask,
    use_candidate_as_base: bool = False,
):
    anchor_box = anchor_mask.box
    assert anchor_box

    candidate_box = candidate_mask.box
    assert candidate_box

    # Calculate intersection.
    up = max(anchor_box.up, candidate_box.up)
    down = min(anchor_box.down, candidate_box.down)
    left = max(anchor_box.left, candidate_box.left)
    right = min(anchor_box.right, candidate_box.right)

    if up > down or left > right:
        return 0.0

    np_intersected_anchor_mask = anchor_mask.mat[
        up - anchor_box.up:down - anchor_box.up + 1,
        left - anchor_box.left:right - anchor_box.left + 1,
    ]  # yapf: disable
    np_intersected_candidate_mask = candidate_mask.mat[
        up - candidate_box.up:down - candidate_box.up + 1,
        left - candidate_box.left:right - candidate_box.left + 1,
    ]  # yapf: disable
    np_intersected_mask = np_intersected_anchor_mask & np_intersected_candidate_mask
    intersected_area = int(np_intersected_mask.sum())

    if use_candidate_as_base:
        base_area = int(candidate_mask.np_mask.sum())
    else:
        base_area = (
            int(anchor_mask.np_mask.sum()) + int(candidate_mask.np_mask.sum()) - intersected_area
        )

    return intersected_area / base_area


class TextRegionFlattener:

    @classmethod
    def patch_text_region_polygons(
        cls,
        text_region_polygons: Sequence[Polygon],
        grouped_char_polygons: Optional[Sequence[Sequence[Polygon]]],
    ):
        if grouped_char_polygons is None:
            return text_region_polygons

        assert len(text_region_polygons) == len(grouped_char_polygons)

        patched_text_region_polygons: List[Polygon] = []
        for text_region_polygon, char_polygons in zip(text_region_polygons, grouped_char_polygons):
            # Need to make sure all char polygons are included.
            unionized_polygons = [text_region_polygon]
            unionized_polygons.extend(char_polygons)

            bounding_box = Box.from_boxes((polygon.bounding_box for polygon in unionized_polygons))
            mask = Mask.from_shapable(bounding_box).to_box_attached(bounding_box)
            for polygon in unionized_polygons:
                polygon.fill_mask(mask)

            patched_text_region_polygons.append(mask.to_external_polygon())

        return patched_text_region_polygons

    @classmethod
    def process_text_region_polygons(
        cls,
        text_region_polygon_dilate_ratio: float,
        shape: Tuple[int, int],
        text_region_polygons: Sequence[Polygon],
        force_no_dilation_flags: Optional[Sequence[bool]] = None,
    ):
        text_mask = Mask.from_polygons(shape, text_region_polygons)
        non_text_mask = text_mask.to_inverted_mask()

        box = Box.from_shape(shape)
        text_mask = text_mask.to_box_attached(box)
        non_text_mask = non_text_mask.to_box_attached(box)

        bounding_extended_text_region_masks: List[Mask] = []
        bounding_rectangular_polygons: List[Polygon] = []

        if force_no_dilation_flags is None:
            force_no_dilation_flags_iter = itertools.repeat(False)
        else:
            assert len(force_no_dilation_flags) == len(text_region_polygons)
            force_no_dilation_flags_iter = force_no_dilation_flags

        for text_region_polygon, force_no_dilation_flag in zip(
            text_region_polygons, force_no_dilation_flags_iter
        ):
            original_text_region_polygon = text_region_polygon

            if not force_no_dilation_flag:
                # Dilate.
                text_region_polygon = text_region_polygon.to_dilated_polygon(
                    ratio=text_region_polygon_dilate_ratio,
                )
                text_region_polygon = text_region_polygon.to_clipped_polygon(shape)

            # Get bounding rectangular box (polygon).
            bounding_rectangular_polygon = \
                text_region_polygon.to_bounding_rectangular_polygon(shape)

            bounding_box = bounding_rectangular_polygon.bounding_box

            # Get other text region.
            bounding_other_text_mask = bounding_rectangular_polygon.extract_mask(text_mask).copy()
            # NOTE: Use the original text region polygon to unset the current text mask.
            original_text_region_polygon.fill_mask(bounding_other_text_mask, 0)

            # Get protentially dilated text region.
            bounding_text_mask = Mask.from_shapable(bounding_other_text_mask)
            bounding_text_mask = bounding_text_mask.to_box_attached(bounding_box)
            # NOTE: Use the protentially dilated text region polygon to set the current text mask.
            text_region_polygon.fill_mask(bounding_text_mask, value=1)

            # Should not use the protentially dilated text region polygon anymore.
            del text_region_polygon

            # Trim protentially dilated text region polygon by eliminating other text region.
            bounding_trimmed_text_mask = Mask.from_masks(
                bounding_box,
                [
                    # Includes the protentially dilated text region.
                    bounding_text_mask,
                    # But not includes any other text regions.
                    bounding_other_text_mask.to_inverted_mask(),
                ],
                ElementSetOperationMode.INTERSECT,
            )

            # Get non-text region.
            bounding_non_text_mask = bounding_rectangular_polygon.extract_mask(non_text_mask)

            # Combine trimmed text region and non-text region.
            bounding_extended_text_region_mask = Mask.from_masks(
                bounding_box,
                [bounding_trimmed_text_mask, bounding_non_text_mask],
            )

            bounding_extended_text_region_masks.append(bounding_extended_text_region_mask)
            bounding_rectangular_polygons.append(bounding_rectangular_polygon)

        return bounding_extended_text_region_masks, bounding_rectangular_polygons

    @classmethod
    def analyze_bounding_rectangular_polygons(
        cls,
        bounding_rectangular_polygons: Sequence[Polygon],
    ):
        long_side_ratios: List[float] = []
        long_side_angles: List[int] = []

        for polygon in bounding_rectangular_polygons:
            # Get reference line.
            point0, point1, _, point3 = polygon.points
            side0_length = math.hypot(
                point0.smooth_y - point1.smooth_y,
                point0.smooth_x - point1.smooth_x,
            )
            side1_length = math.hypot(
                point0.smooth_y - point3.smooth_y,
                point0.smooth_x - point3.smooth_x,
            )

            long_side_ratios.append(
                max(side0_length, side1_length) / min(side0_length, side1_length)
            )

            point_a = point0
            if side0_length > side1_length:
                # Reference line (p0 -> p1).
                point_b = point1
            else:
                # Reference line (p0 -> p3).
                point_b = point3

            # Get the angle of reference line, in [0, 180) degree.
            np_theta = np.arctan2(
                point_a.smooth_y - point_b.smooth_y,
                point_a.smooth_x - point_b.smooth_x,
            )
            np_theta = np_theta % np.pi
            long_side_angle = round(np_theta / np.pi * 180) % 180
            long_side_angles.append(long_side_angle)

        return long_side_ratios, long_side_angles

    @classmethod
    def get_typical_angle(
        cls,
        typical_long_side_ratio_min: float,
        long_side_ratios: Sequence[float],
        long_side_angles: Sequence[int],
    ):
        typical_indices: Set[int] = set()
        typical_long_side_angles: List[float] = []

        for idx, (long_side_ratio, long_side_angle) in \
                enumerate(zip(long_side_ratios, long_side_angles)):
            if long_side_ratio < typical_long_side_ratio_min:
                continue

            typical_indices.add(idx)
            typical_long_side_angles.append(long_side_angle)

        if not typical_long_side_angles:
            return None, typical_indices

        # NOTE: Due to the sudden change between 179 and 0 degree,
        # we need to normalize the range to [0, 360) before calculate the mean of angles.
        two_pi = 2 * np.pi
        np_angles = np.asarray(typical_long_side_angles) / 180 * two_pi
        np_sin_mean = np.sin(np_angles).mean()
        np_cos_mean = np.cos(np_angles).mean()

        np_theta = np.arctan2(np_sin_mean, np_cos_mean)
        np_theta = np_theta % two_pi
        # Rescale the range back to [0, 180).
        typical_angle = round(np_theta / two_pi * 180)

        return typical_angle, typical_indices

    @classmethod
    def get_flattening_rotate_angles(
        cls,
        typical_angle: Optional[int],
        typical_indices: Set[int],
        long_side_angles: Sequence[int],
    ):
        if typical_angle is not None:
            assert typical_indices

        flattening_rotate_angles: List[int] = []

        for idx, long_side_angle in enumerate(long_side_angles):
            if typical_angle is None or idx in typical_indices:
                # Dominated by long_side_angle.
                main_angle = long_side_angle

            else:
                # Dominated by typical_angle.
                short_side_angle = (long_side_angle + 90) % 180
                long_side_delta = abs((long_side_angle - typical_angle + 90) % 180 - 90)
                short_side_delta = abs((short_side_angle - typical_angle + 90) % 180 - 90)

                if long_side_delta < short_side_delta:
                    main_angle = long_side_angle
                else:
                    main_angle = short_side_angle

            # Angle for flattening.
            if main_angle <= 90:
                # [270, 360).
                flattening_rotate_angle = (360 - main_angle) % 360
            else:
                # [1, 90).
                flattening_rotate_angle = 180 - main_angle
            flattening_rotate_angles.append(flattening_rotate_angle)

        return flattening_rotate_angles

    @classmethod
    def build_flattened_text_regions(
        cls,
        image: Image,
        text_region_polygons: Sequence[Polygon],
        bounding_extended_text_region_masks: Sequence[Mask],
        typical_indices: Set[int],
        flattening_rotate_angles: Sequence[int],
        grouped_char_polygons: Optional[Sequence[Sequence[Polygon]]],
    ):
        flattened_text_regions: List[FlattenedTextRegion] = []

        for idx, (
            text_region_polygon,
            bounding_extended_text_region_mask,
            flattening_rotate_angle,
        ) in enumerate(
            zip(
                text_region_polygons,
                bounding_extended_text_region_masks,
                flattening_rotate_angles,
            )
        ):
            bounding_box = bounding_extended_text_region_mask.box
            assert bounding_box

            # Extract image.
            text_region_image = bounding_extended_text_region_mask.extract_image(image)

            # Shift char polygons.
            relative_char_polygons = None
            if grouped_char_polygons is not None:
                char_polygons = grouped_char_polygons[idx]
                relative_char_polygons = [
                    char_polygon.to_relative_polygon(
                        origin_y=bounding_box.up,
                        origin_x=bounding_box.left,
                    ) for char_polygon in char_polygons
                ]

            # Rotate.
            rotated_result = rotate.distort(
                {'angle': flattening_rotate_angle},
                image=text_region_image,
                mask=bounding_extended_text_region_mask,
                polygons=relative_char_polygons,
            )
            rotated_text_region_image = rotated_result.image
            assert rotated_text_region_image
            rotated_bounding_extended_text_region_mask = rotated_result.mask
            assert rotated_bounding_extended_text_region_mask
            # Could be None.
            rotated_char_polygons = rotated_result.polygons

            # Trim.
            rotated_trimmed_box = rotated_bounding_extended_text_region_mask.to_external_box()

            trimmed_text_region_image = rotated_text_region_image.to_cropped_image(
                up=rotated_trimmed_box.up,
                down=rotated_trimmed_box.down,
                left=rotated_trimmed_box.left,
                right=rotated_trimmed_box.right,
            )

            trimmed_mask = rotated_trimmed_box.extract_mask(
                rotated_bounding_extended_text_region_mask
            )

            trimmed_char_polygons = None
            if rotated_char_polygons:
                trimmed_char_polygons = [
                    rotated_char_polygon.to_relative_polygon(
                        origin_y=rotated_trimmed_box.up,
                        origin_x=rotated_trimmed_box.left,
                    ) for rotated_char_polygon in rotated_char_polygons
                ]

            flattened_text_regions.append(
                FlattenedTextRegion(
                    is_typical=(idx in typical_indices),
                    text_region_polygon=text_region_polygon,
                    text_region_image=bounding_extended_text_region_mask.extract_image(image),
                    bounding_extended_text_region_mask=bounding_extended_text_region_mask,
                    flattening_rotate_angle=flattening_rotate_angle,
                    shape_before_trim=rotated_text_region_image.shape,
                    rotated_trimmed_box=rotated_trimmed_box,
                    shape_before_resize=trimmed_text_region_image.shape,
                    post_rotate_angle=0,
                    flattened_image=trimmed_text_region_image,
                    flattened_mask=trimmed_mask,
                    flattened_char_polygons=trimmed_char_polygons,
                )
            )

        return flattened_text_regions

    def __init__(
        self,
        typical_long_side_ratio_min: float,
        text_region_polygon_dilate_ratio: float,
        image: Image,
        text_region_polygons: Sequence[Polygon],
        grouped_char_polygons: Optional[Sequence[Sequence[Polygon]]] = None,
        is_training: bool = False,
    ):
        self.origional_text_region_polygons = text_region_polygons

        self.text_region_polygons = self.patch_text_region_polygons(
            text_region_polygons=text_region_polygons,
            grouped_char_polygons=grouped_char_polygons,
        )

        force_no_dilation_flags = None
        if is_training:
            assert grouped_char_polygons and len(text_region_polygons) == len(grouped_char_polygons)
            force_no_dilation_flags = []
            for char_polygons in grouped_char_polygons:
                force_no_dilation_flags.append(not char_polygons)

        self.bounding_extended_text_region_masks, self.bounding_rectangular_polygons = \
            self.process_text_region_polygons(
                text_region_polygon_dilate_ratio=text_region_polygon_dilate_ratio,
                shape=image.shape,
                text_region_polygons=self.text_region_polygons,
                force_no_dilation_flags=force_no_dilation_flags,
            )

        self.long_side_ratios, self.long_side_angles = \
            self.analyze_bounding_rectangular_polygons(self.bounding_rectangular_polygons)

        self.typical_angle, self.typical_indices = self.get_typical_angle(
            typical_long_side_ratio_min=typical_long_side_ratio_min,
            long_side_ratios=self.long_side_ratios,
            long_side_angles=self.long_side_angles,
        )

        self.flattening_rotate_angles = self.get_flattening_rotate_angles(
            typical_angle=self.typical_angle,
            typical_indices=self.typical_indices,
            long_side_angles=self.long_side_angles,
        )

        self.flattened_text_regions = self.build_flattened_text_regions(
            image=image,
            text_region_polygons=self.origional_text_region_polygons,
            bounding_extended_text_region_masks=self.bounding_extended_text_region_masks,
            typical_indices=self.typical_indices,
            flattening_rotate_angles=self.flattening_rotate_angles,
            grouped_char_polygons=grouped_char_polygons,
        )


def build_background_image_for_stacking(height: int, width: int):
    np_rgb_rows = [np.zeros((width, 3), dtype=np.uint8) for _ in range(3)]
    rgb_tuples = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for color_offset, np_row in enumerate(np_rgb_rows):
        for color_idx in range(3):
            color_tuple = rgb_tuples[(color_offset + color_idx) % 3]
            np_row[color_idx::3] = color_tuple

    np_image = np.zeros((height, width, 3), dtype=np.uint8)
    for row_offset, np_row in enumerate(np_rgb_rows):
        np_image[row_offset::3] = np_row

    return Image(mat=np_image)


def stack_flattened_text_regions(
    page_pad: int,
    flattened_text_regions_pad: int,
    flattened_text_regions: Sequence[FlattenedTextRegion],
):
    page_double_pad = 2 * page_pad
    flattened_text_regions_double_pad = 2 * flattened_text_regions_pad

    rect_packer = RectPacker(rotation=False)

    # Add box and bin.
    # NOTE: Only one bin is added, that is, packing all text region into one image.
    bin_width = 0
    bin_height = 0

    for ftr_idx, flattened_text_region in enumerate(flattened_text_regions):
        rect_packer.add_rect(
            width=flattened_text_region.width + flattened_text_regions_double_pad,
            height=flattened_text_region.height + flattened_text_regions_double_pad,
            rid=ftr_idx,
        )

        bin_width = max(bin_width, flattened_text_region.width)
        bin_height += flattened_text_region.height

    bin_width += flattened_text_regions_double_pad
    bin_height += flattened_text_regions_double_pad

    rect_packer.add_bin(width=bin_width, height=bin_height)

    # Pack boxes.
    rect_packer.pack()  # type: ignore

    # Get packed boxes.
    unordered_boxes: List[Box] = []
    ftr_indices: List[int] = []
    for bin_idx, x, y, width, height, ftr_idx in rect_packer.rect_list():
        assert bin_idx == 0
        unordered_boxes.append(Box(
            up=y,
            down=y + height - 1,
            left=x,
            right=x + width - 1,
        ))
        ftr_indices.append(ftr_idx)

    # Order boxes.
    inverse_ftr_indices = [-1] * len(ftr_indices)
    for inverse_ftr_idx, ftr_idx in enumerate(ftr_indices):
        inverse_ftr_indices[ftr_idx] = inverse_ftr_idx
    for inverse_ftr_idx in inverse_ftr_indices:
        assert inverse_ftr_idx >= 0
    padded_boxes = [unordered_boxes[inverse_ftr_idx] for inverse_ftr_idx in inverse_ftr_indices]

    page_height = max(box.down for box in padded_boxes) + 1 + page_double_pad
    page_width = max(box.right for box in padded_boxes) + 1 + page_double_pad

    image = build_background_image_for_stacking(page_height, page_width)
    boxes: List[Box] = []
    char_polygons: List[Polygon] = []

    for padded_box, flattened_text_region in zip(padded_boxes, flattened_text_regions):
        assert flattened_text_region.height + flattened_text_regions_double_pad \
            == padded_box.height
        assert flattened_text_region.width + flattened_text_regions_double_pad \
            == padded_box.width

        # Remove box padding.
        up = padded_box.up + flattened_text_regions_pad + page_pad
        left = padded_box.left + flattened_text_regions_pad + page_pad

        box = Box(
            up=up,
            down=up + flattened_text_region.height - 1,
            left=left,
            right=left + flattened_text_region.width - 1,
        )
        boxes.append(box)

        # Render.
        box.fill_image(
            image,
            flattened_text_region.flattened_image,
            image_mask=flattened_text_region.flattened_mask,
        )

        if flattened_text_region.flattened_char_polygons:
            for char_polygon in flattened_text_region.flattened_char_polygons:
                char_polygons.append(char_polygon.to_shifted_polygon(
                    offset_y=up,
                    offset_x=left,
                ))

    return image, boxes, char_polygons


class PageTextRegionStep(
    PipelineStep[
        PageTextRegionStepConfig,
        PageTextRegionStepInput,
        PageTextRegionStepOutput,
    ]
):  # yapf: disable

    @classmethod
    def generate_precise_text_region_candidate_polygons(
        cls,
        precise_mask: Mask,
        disconnected_text_region_mask: Mask,
    ):
        assert precise_mask.box and disconnected_text_region_mask.box

        # Get the intersection.
        intersected_box = Box(
            up=max(precise_mask.box.up, disconnected_text_region_mask.box.up),
            down=min(precise_mask.box.down, disconnected_text_region_mask.box.down),
            left=max(precise_mask.box.left, disconnected_text_region_mask.box.left),
            right=min(precise_mask.box.right, disconnected_text_region_mask.box.right),
        )
        assert intersected_box.up <= intersected_box.down
        assert intersected_box.left <= intersected_box.right

        precise_mask = intersected_box.extract_mask(precise_mask)
        disconnected_text_region_mask = intersected_box.extract_mask(disconnected_text_region_mask)

        # Apply mask bitwise-and operation.
        intersected_mask = Mask(
            mat=(disconnected_text_region_mask.mat & precise_mask.mat).astype(np.uint8)
        )
        intersected_mask = intersected_mask.to_box_attached(intersected_box)

        # NOTE:
        # 1. Could extract more than one polygons.
        # 2. Some polygons are in border and should be removed later.
        return intersected_mask.to_disconnected_polygons()

    @classmethod
    def strtree_query_intersected_polygons(
        cls,
        strtree: STRtree,
        id_to_anchor_polygon: Dict[int, Polygon],
        candidate_polygon: Polygon,
    ):
        candidate_shapely_polygon = candidate_polygon.to_shapely_polygon()
        candidate_mask = candidate_polygon.mask

        for anchor_shapely_polygon in strtree.query(candidate_shapely_polygon):
            anchor_id = id(anchor_shapely_polygon)
            anchor_polygon = id_to_anchor_polygon[anchor_id]
            anchor_mask = anchor_polygon.mask

            intersected_ratio = calculate_boxed_masks_intersected_ratio(
                anchor_mask=anchor_mask,
                candidate_mask=candidate_mask,
                use_candidate_as_base=True,
            )

            yield (
                anchor_id,
                anchor_polygon,
                anchor_mask,
                candidate_mask,
                intersected_ratio,
            )

    def sample_page_non_text_region_polygons(
        self,
        page_non_text_region_polygons: Sequence[Polygon],
        num_page_text_region_infos: int,
        rng: RandomGenerator,
    ):
        negative_ratio = self.config.negative_text_region_ratio
        num_page_non_text_region_polygons = round(
            negative_ratio * num_page_text_region_infos / (1 - negative_ratio)
        )
        return rng_choice_with_size(
            rng,
            page_non_text_region_polygons,
            size=min(
                num_page_non_text_region_polygons,
                len(page_non_text_region_polygons),
            ),
            replace=False,
        )

    def build_flattened_text_regions(
        self,
        page_image: Image,
        page_text_region_infos: Sequence[PageTextRegionInfo],
        page_non_text_region_polygons: Sequence[Polygon],
        rng: RandomGenerator,
    ):
        text_region_polygon_dilate_ratio = float(
            rng.uniform(
                self.config.text_region_flattener_text_region_polygon_dilate_ratio_min,
                self.config.text_region_flattener_text_region_polygon_dilate_ratio_max,
            )
        )
        typical_long_side_ratio_min = \
            self.config.text_region_flattener_typical_long_side_ratio_min

        text_region_polygons: List[Polygon] = []
        grouped_char_polygons: List[Sequence[Polygon]] = []
        for page_text_region_info in page_text_region_infos:
            text_region_polygons.append(page_text_region_info.precise_text_region_polygon)
            grouped_char_polygons.append(page_text_region_info.char_polygons)

        # Inject nagative regions.
        for page_non_text_region_polygon in page_non_text_region_polygons:
            text_region_polygons.append(page_non_text_region_polygon)
            grouped_char_polygons.append(tuple())

        text_region_flattener = TextRegionFlattener(
            typical_long_side_ratio_min=typical_long_side_ratio_min,
            text_region_polygon_dilate_ratio=text_region_polygon_dilate_ratio,
            image=page_image,
            text_region_polygons=text_region_polygons,
            grouped_char_polygons=grouped_char_polygons,
            is_training=True,
        )

        # Resize positive ftr.
        positive_flattened_text_regions: List[FlattenedTextRegion] = []
        # For negative sampling.
        positive_reference_heights: List[float] = []
        positive_reference_widths: List[float] = []
        num_negative_flattened_text_regions = 0

        for flattened_text_region in text_region_flattener.flattened_text_regions:
            if not flattened_text_region.flattened_char_polygons:
                num_negative_flattened_text_regions += 1
                continue

            char_height_median = flattened_text_region.get_char_height_meidan()

            text_region_resize_char_height_median = int(
                rng.integers(
                    self.config.text_region_resize_char_height_median_min,
                    self.config.text_region_resize_char_height_median_max + 1,
                )
            )
            scale = text_region_resize_char_height_median / char_height_median

            height, width = flattened_text_region.shape
            resized_height = round(height * scale)
            resized_width = round(width * scale)

            flattened_text_region = flattened_text_region.to_resized_flattened_text_region(
                resized_height=resized_height,
                resized_width=resized_width,
            )

            positive_reference_heights.append(resized_height)
            positive_reference_widths.append(resized_width)

            # Post rotate.
            post_rotate_angle = 0
            if flattened_text_region.is_typical:
                if rng.random() < self.config.text_region_typical_post_rotate_prob:
                    # Upside down only.
                    post_rotate_angle = 180
            else:
                if rng.random() < self.config.text_region_untypical_post_rotate_prob:
                    # 3-way rotate.
                    post_rotate_angle = rng_choice(rng, (180, 90, 270), probs=(0.5, 0.25, 0.25))

            if post_rotate_angle != 0:
                flattened_text_region = \
                    flattened_text_region.to_post_rotated_flattened_text_region(post_rotate_angle)

            positive_flattened_text_regions.append(flattened_text_region)

        # Resize negative ftr.
        negative_reference_heights = list(
            rng_choice_with_size(
                rng,
                positive_reference_heights,
                size=num_negative_flattened_text_regions,
                replace=(num_negative_flattened_text_regions > len(positive_reference_heights)),
            )
        )

        negative_height_max = max(positive_reference_heights)
        negative_width_max = max(positive_reference_widths)

        negative_flattened_text_regions: List[FlattenedTextRegion] = []

        for flattened_text_region in text_region_flattener.flattened_text_regions:
            if flattened_text_region.flattened_char_polygons:
                continue

            reference_height = negative_reference_heights.pop()
            scale = reference_height / flattened_text_region.height

            height, width = flattened_text_region.shape
            resized_height = round(height * scale)
            resized_width = round(width * scale)

            # Remove negative region that is too large.
            if resized_height > negative_height_max or resized_width > negative_width_max:
                continue

            flattened_text_region = flattened_text_region.to_resized_flattened_text_region(
                resized_height=resized_height,
                resized_width=resized_width,
            )

            # Post rotate.
            post_rotate_angle = 0
            if flattened_text_region.is_typical:
                if rng.random() < self.config.text_region_typical_post_rotate_prob:
                    # Upside down only.
                    post_rotate_angle = 180
            else:
                if rng.random() < self.config.text_region_untypical_post_rotate_prob:
                    # 3-way rotate.
                    post_rotate_angle = rng_choice(rng, (180, 90, 270), probs=(0.5, 0.25, 0.25))

            if post_rotate_angle != 0:
                flattened_text_region = \
                    flattened_text_region.to_post_rotated_flattened_text_region(post_rotate_angle)

            negative_flattened_text_regions.append(flattened_text_region)

        flattened_text_regions = (
            *positive_flattened_text_regions,
            *negative_flattened_text_regions,
        )
        return flattened_text_regions

    def run(self, input: PageTextRegionStepInput, rng: RandomGenerator):
        page_distortion_step_output = input.page_distortion_step_output
        page_image = page_distortion_step_output.page_image
        page_char_polygon_collection = page_distortion_step_output.page_char_polygon_collection
        page_disconnected_text_region_collection = \
            page_distortion_step_output.page_disconnected_text_region_collection
        page_non_text_region_collection = \
            page_distortion_step_output.page_non_text_region_collection

        page_resizing_step_output = input.page_resizing_step_output
        page_resized_text_line_mask = page_resizing_step_output.page_text_line_mask

        debug = None
        if self.config.enable_debug:
            debug = PageTextRegionStepDebug()

        # Build R-tree to track text regions.
        # https://github.com/shapely/shapely/issues/640
        id_to_disconnected_text_region_polygon: Dict[int, Polygon] = {}
        disconnected_text_region_shapely_polygons: List[ShapelyPolygon] = []

        for polygon in page_disconnected_text_region_collection.to_polygons():
            shapely_polygon = polygon.to_shapely_polygon()
            id_to_disconnected_text_region_polygon[id(shapely_polygon)] = polygon
            disconnected_text_region_shapely_polygons.append(shapely_polygon)

        disconnected_text_region_tree = STRtree(disconnected_text_region_shapely_polygons)

        # Get the precise text regions.
        precise_text_region_candidate_polygons: List[Polygon] = []
        for resized_precise_polygon in page_resized_text_line_mask.to_disconnected_polygons():
            # Resize back to the shape after distortion.
            precise_polygon = resized_precise_polygon.to_conducted_resized_polygon(
                page_resized_text_line_mask,
                resized_height=page_image.height,
                resized_width=page_image.width,
            )

            # Find and extract intersected text region.
            # NOTE: One precise_polygon could be overlapped with
            # more than one disconnected_text_region_polygon!
            for _, _, disconnected_text_region_mask, precise_mask, _ in \
                    self.strtree_query_intersected_polygons(
                        strtree=disconnected_text_region_tree,
                        id_to_anchor_polygon=id_to_disconnected_text_region_polygon,
                        candidate_polygon=precise_polygon,
                    ):
                precise_text_region_candidate_polygons.extend(
                    self.generate_precise_text_region_candidate_polygons(
                        precise_mask=precise_mask,
                        disconnected_text_region_mask=disconnected_text_region_mask,
                    )
                )

        if debug:
            debug.page_image = page_image
            debug.precise_text_region_candidate_polygons = precise_text_region_candidate_polygons

        # Help gc.
        del id_to_disconnected_text_region_polygon
        del disconnected_text_region_shapely_polygons
        del disconnected_text_region_tree

        # Bind char-level polygon to precise text region.
        id_to_precise_text_region_polygon: Dict[int, Polygon] = {}
        precise_text_region_shapely_polygons: List[ShapelyPolygon] = []

        for polygon in precise_text_region_candidate_polygons:
            shapely_polygon = polygon.to_shapely_polygon()
            id_to_precise_text_region_polygon[id(shapely_polygon)] = polygon
            precise_text_region_shapely_polygons.append(shapely_polygon)

        precise_text_region_tree = STRtree(precise_text_region_shapely_polygons)

        id_to_char_polygons: DefaultDict[int, List[Polygon]] = defaultdict(list)
        for char_polygon in page_char_polygon_collection.polygons:
            best_precise_text_region_id = None
            intersected_ratio_max = 0

            for (
                precise_text_region_id,
                _,
                _,
                _,
                intersected_ratio,
            ) in self.strtree_query_intersected_polygons(
                strtree=precise_text_region_tree,
                id_to_anchor_polygon=id_to_precise_text_region_polygon,
                candidate_polygon=char_polygon,
            ):
                if intersected_ratio > intersected_ratio_max:
                    intersected_ratio_max = intersected_ratio
                    best_precise_text_region_id = precise_text_region_id

            if best_precise_text_region_id is not None:
                id_to_char_polygons[best_precise_text_region_id].append(char_polygon)
            else:
                # NOTE: Text line with only a small char (i.e. delimiter) could enter this branch.
                # In such case, the text line bounding box is smaller than the char polygon, since
                # the leading/trailing char paddings are ignored during text line rendering.
                # It's acceptable for now since: 1) this case happens rarely, 2) and it won't
                # introduce labeling noise.
                logger.warning(f'Cannot assign a text region for char_polygon={char_polygon}')

        page_text_region_infos: List[PageTextRegionInfo] = []
        for precise_text_region_shapely_polygon in precise_text_region_shapely_polygons:
            ptrsp_id = id(precise_text_region_shapely_polygon)
            if ptrsp_id not in id_to_char_polygons:
                # Not related to any char polygons.
                continue
            assert id_to_char_polygons[ptrsp_id]
            page_text_region_infos.append(
                PageTextRegionInfo(
                    precise_text_region_polygon=id_to_precise_text_region_polygon[ptrsp_id],
                    char_polygons=id_to_char_polygons[ptrsp_id],
                )
            )

        # Help gc.
        del id_to_precise_text_region_polygon
        del precise_text_region_shapely_polygons
        del precise_text_region_tree

        if debug:
            debug.page_text_region_infos = page_text_region_infos

        # Negative sampling.
        page_non_text_region_polygons = self.sample_page_non_text_region_polygons(
            page_non_text_region_polygons=tuple(page_non_text_region_collection.to_polygons()),
            num_page_text_region_infos=len(page_text_region_infos),
            rng=rng,
        )

        flattened_text_regions = self.build_flattened_text_regions(
            page_image=page_image,
            page_text_region_infos=page_text_region_infos,
            page_non_text_region_polygons=page_non_text_region_polygons,
            rng=rng,
        )
        if debug:
            debug.flattened_text_regions = flattened_text_regions

        # Stack text regions.
        image, _, char_polygons = stack_flattened_text_regions(
            page_pad=0,
            flattened_text_regions_pad=self.config.stack_flattened_text_regions_pad,
            flattened_text_regions=flattened_text_regions,
        )

        # Post uniform rotation.
        shape_before_rotate = image.shape
        rotate_angle = 0

        if self.config.enable_post_rotate:
            rotate_angle = int(
                rng.integers(
                    self.config.post_rotate_angle_min,
                    self.config.post_rotate_angle_max + 1,
                )
            )
            rotated_result = rotate.distort(
                {'angle': rotate_angle},
                image=image,
                polygons=char_polygons,
            )
            assert rotated_result.image and rotated_result.polygons
            image = rotated_result.image
            char_polygons = rotated_result.polygons

        return PageTextRegionStepOutput(
            page_image=image,
            page_char_polygons=char_polygons,
            shape_before_rotate=shape_before_rotate,
            rotate_angle=rotate_angle,
            debug=debug,
        )


page_text_region_step_factory = PipelineStepFactory(PageTextRegionStep)
