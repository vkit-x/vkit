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
import warnings

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
from shapely.errors import ShapelyDeprecationWarning
from shapely.strtree import STRtree
from shapely.geometry import Polygon as ShapelyPolygon
from rectpack import newPacker as RectPacker

from vkit.element import Box, Polygon, Mask, Image
from vkit.engine.distortion.geometric.affine import rotate
from ..interface import PipelineStep, PipelineStepFactory
from .page_distortion import PageDistortionStepOutput
from .page_resizing import PageResizingStepOutput

# Shapely version has been explicitly locked under 2.0, hence ignore this warning.
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)


@attrs.define
class PageTextRegionStepConfig:
    page_flat_text_region_resize_char_height_min: int = 32
    page_flat_text_region_resize_char_height_max: int = 36
    text_region_flattener_typical_long_side_ratio_min: float = 3.0
    text_region_flattener_text_region_polygon_dilate_ratio_min: float = 0.9
    text_region_flattener_text_region_polygon_dilate_ratio_max: float = 1.0
    gap: int = 2
    enable_post_uniform_rotate: bool = False
    debug: bool = False


@attrs.define
class PageTextRegionStepInput:
    page_distortion_step_output: PageDistortionStepOutput
    page_resizing_step_output: PageResizingStepOutput


@attrs.define
class PageTextRegionInfo:
    precise_text_region_polygon: Polygon
    char_polygons: Sequence[Polygon]


@attrs.define
class PageFlatTextRegion:
    image: Image
    char_polygons: Sequence[Polygon]

    @property
    def height(self):
        return self.image.height

    @property
    def width(self):
        return self.image.width


@attrs.define
class DebugPageTextRegionStep:
    page_image: Image = attrs.field(default=None)
    precise_text_region_candidate_polygons: List[Polygon] = attrs.field(default=None)
    page_text_region_infos: List[PageTextRegionInfo] = attrs.field(default=None)
    page_flat_text_regions: List[PageFlatTextRegion] = attrs.field(default=None)
    resized_page_flat_text_regions: List[PageFlatTextRegion] = attrs.field(default=None)


@attrs.define
class PageTextRegionStepOutput:
    page_image: Image
    page_char_polygons: Sequence[Polygon]
    shape_before_rotate: Tuple[int, int]
    rotate_angle: int
    debug: Optional[DebugPageTextRegionStep]


@attrs.define
class FlattenedTextRegion:
    text_region_polygon: Polygon
    flattening_rotate_angle: int
    trim_up: int
    trim_left: int
    flattened_image: Image
    flattened_char_polygons: Optional[Sequence[Polygon]]


class TextRegionFlattener:

    @staticmethod
    def patch_text_region_polygons(
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

    @staticmethod
    def build_minimum_rotated_rectangles(text_region_polygons: Sequence[Polygon]):
        minimum_rotated_rectangles: List[Polygon] = []

        for text_region_polygon in text_region_polygons:
            # Get minimum bounding rectangle.
            shapely_polygon = text_region_polygon.to_shapely_polygon()
            minimum_rotated_rectangle = shapely_polygon.minimum_rotated_rectangle

            assert isinstance(minimum_rotated_rectangle, ShapelyPolygon)
            polygon = Polygon.from_shapely_polygon(minimum_rotated_rectangle)
            assert len(polygon.points) == 4

            minimum_rotated_rectangles.append(polygon)

        return minimum_rotated_rectangles

    @staticmethod
    def analyze_build_minimum_rotated_rectangles(minimum_rotated_rectangles: Sequence[Polygon]):
        long_side_ratios: List[float] = []
        long_side_angles: List[int] = []

        for polygon in minimum_rotated_rectangles:
            # Get reference line.
            point0, point1, _, point3 = polygon.points
            side0_length = math.hypot(point0.y - point1.y, point0.x - point1.x)
            side1_length = math.hypot(point0.y - point3.y, point0.x - point3.x)

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

            # Get angle of reference line in [0, 180)
            np_theta = np.arctan2(
                point_a.y - point_b.y,
                point_a.x - point_b.x,
            )
            np_theta = np_theta % np.pi
            angle = round(np_theta / np.pi * 180)

            long_side_angles.append(angle)

        return long_side_ratios, long_side_angles

    @staticmethod
    def get_typical_angle(
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

        np_angles = np.asarray(typical_long_side_angles) / 180 * np.pi
        np_sin_mean = np.sin(np_angles).mean()
        np_cos_mean = np.cos(np_angles).mean()

        np_theta = np.arctan2(np_sin_mean, np_cos_mean)
        np_theta = np_theta % np.pi
        typical_angle = round(np_theta / np.pi * 180)

        return typical_angle, typical_indices

    @staticmethod
    def get_flattening_rotate_angles(
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
                flattening_rotate_angle = 360 - main_angle
            else:
                flattening_rotate_angle = 180 - main_angle
            flattening_rotate_angles.append(flattening_rotate_angle)

        return flattening_rotate_angles

    @staticmethod
    def build_flattened_text_regions(
        text_region_polygon_dilate_ratio: float,
        image: Image,
        text_region_polygons: Sequence[Polygon],
        flattening_rotate_angles: Sequence[int],
        grouped_char_polygons: Optional[Sequence[Sequence[Polygon]]],
    ):
        flattened_text_regions: List[FlattenedTextRegion] = []

        for idx, (text_region_polygon, flattening_rotate_angle) in enumerate(
            zip(text_region_polygons, flattening_rotate_angles)
        ):
            # Dilate.
            text_region_polygon = text_region_polygon.to_dilated_polygon(
                text_region_polygon_dilate_ratio
            )
            text_region_polygon = text_region_polygon.to_clipped_polygon(image)

            # Extract image.
            text_region_image = text_region_polygon.extract_image(image)

            # Shift char polygons.
            relative_char_polygons = None
            if grouped_char_polygons is not None:
                char_polygons = grouped_char_polygons[idx]
                relative_char_polygons = [
                    char_polygon.to_relative_polygon(
                        origin_y=text_region_polygon.bounding_box.up,
                        origin_x=text_region_polygon.bounding_box.left,
                    ) for char_polygon in char_polygons
                ]

            # Rotate.
            rotated_result = rotate.distort(
                {'angle': flattening_rotate_angle},
                image=text_region_image,
                polygon=text_region_polygon.self_relative_polygon,
                polygons=relative_char_polygons,
            )
            rotated_text_region_image = rotated_result.image
            assert rotated_text_region_image
            rotated_self_relative_polygon = rotated_result.polygon
            assert rotated_self_relative_polygon
            # Could be None.
            rotated_char_polygons = rotated_result.polygons

            # Trim.
            rotated_bounding_box = rotated_self_relative_polygon.bounding_box
            trimmed_text_region_image = rotated_text_region_image.to_cropped_image(
                up=rotated_bounding_box.up,
                down=rotated_bounding_box.down,
                left=rotated_bounding_box.left,
                right=rotated_bounding_box.right,
            )
            trim_up = rotated_bounding_box.up
            trim_left = rotated_bounding_box.left

            trimmed_char_polygons = None
            if rotated_char_polygons:
                trimmed_char_polygons = [
                    rotated_char_polygon.to_relative_polygon(
                        origin_y=trim_up,
                        origin_x=trim_left,
                    ) for rotated_char_polygon in rotated_char_polygons
                ]

            flattened_text_regions.append(
                FlattenedTextRegion(
                    text_region_polygon=text_region_polygon,
                    flattening_rotate_angle=flattening_rotate_angle,
                    trim_up=trim_up,
                    trim_left=trim_left,
                    flattened_image=trimmed_text_region_image,
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
    ):
        self.origional_text_region_polygons = text_region_polygons

        self.text_region_polygons = self.patch_text_region_polygons(
            text_region_polygons=text_region_polygons,
            grouped_char_polygons=grouped_char_polygons,
        )

        self.minimum_rotated_rectangles = \
            self.build_minimum_rotated_rectangles(self.text_region_polygons)

        self.long_side_ratios, self.long_side_angles = \
            self.analyze_build_minimum_rotated_rectangles(self.minimum_rotated_rectangles)

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
            text_region_polygon_dilate_ratio=text_region_polygon_dilate_ratio,
            image=image,
            text_region_polygons=self.text_region_polygons,
            flattening_rotate_angles=self.flattening_rotate_angles,
            grouped_char_polygons=grouped_char_polygons,
        )


class PageTextRegionStep(
    PipelineStep[
        PageTextRegionStepConfig,
        PageTextRegionStepInput,
        PageTextRegionStepOutput,
    ]
):  # yapf: disable

    @staticmethod
    def generate_precise_text_region_candidate_polygons(
        precise_mask: Mask,
        text_region_mask: Mask,
    ):
        assert precise_mask.box and text_region_mask.box

        # Get the intersection.
        intersected_box = Box(
            up=max(precise_mask.box.up, text_region_mask.box.up),
            down=min(precise_mask.box.down, text_region_mask.box.down),
            left=max(precise_mask.box.left, text_region_mask.box.left),
            right=min(precise_mask.box.right, text_region_mask.box.right),
        )
        assert intersected_box.up <= intersected_box.down
        assert intersected_box.left <= intersected_box.right

        precise_mask = intersected_box.extract_mask(precise_mask)
        text_region_mask = intersected_box.extract_mask(text_region_mask)

        # Apply mask bitwise-and operation.
        intersected_mask = Mask(mat=(text_region_mask.mat & precise_mask.mat).astype(np.uint8))
        intersected_mask = intersected_mask.to_box_attached(intersected_box)

        # NOTE:
        # 1. Could extract more than one polygons.
        # 2. Some polygons are in border and should be removed later.
        return intersected_mask.to_disconnected_polygons()

    @staticmethod
    def strtree_query_intersected_polygons(
        strtree: STRtree,
        id_to_anchor_polygon: Dict[int, Polygon],
        id_to_anchor_mask: Dict[int, Mask],
        id_to_anchor_np_mask: Dict[int, np.ndarray],
        candidate_polygon: Polygon,
    ):
        candidate_shapely_polygon = candidate_polygon.to_shapely_polygon()
        candidate_mask = candidate_polygon.mask
        candidate_box = candidate_mask.box
        assert candidate_box

        for anchor_shapely_polygon in strtree.query(candidate_shapely_polygon):
            anchor_id = id(anchor_shapely_polygon)
            anchor_polygon = id_to_anchor_polygon[anchor_id]

            # Build mask if not exists.
            if anchor_id not in id_to_anchor_mask:
                id_to_anchor_mask[anchor_id] = anchor_polygon.mask
                id_to_anchor_np_mask[anchor_id] = id_to_anchor_mask[anchor_id].np_mask
            anchor_mask = id_to_anchor_mask[anchor_id]
            anchor_np_mask = id_to_anchor_np_mask[anchor_id]
            anchor_box = anchor_mask.box
            assert anchor_box

            # Calculate intersection.
            intersected_box = Box(
                up=max(anchor_box.up, candidate_box.up),
                down=min(anchor_box.down, candidate_box.down),
                left=max(anchor_box.left, candidate_box.left),
                right=min(anchor_box.right, candidate_box.right),
            )
            # strtree.query is based on envelope overlapping, enhance should be a valid box.
            assert intersected_box.up <= intersected_box.down
            assert intersected_box.left <= intersected_box.right

            # For optimizing performance.
            up = intersected_box.up - anchor_box.up
            down = intersected_box.down - anchor_box.up
            left = intersected_box.left - anchor_box.left
            right = intersected_box.right - anchor_box.left
            np_anchor_mask = anchor_np_mask[up:down + 1, left:right + 1]

            up = intersected_box.up - candidate_box.up
            down = intersected_box.down - candidate_box.up
            left = intersected_box.left - candidate_box.left
            right = intersected_box.right - candidate_box.left
            np_candidate_mask = candidate_mask.np_mask[up:down + 1, left:right + 1]

            np_intersected_mask = (np_anchor_mask & np_candidate_mask)
            intersected_area = int(np_intersected_mask.sum())
            if intersected_area == 0:
                continue

            yield (
                anchor_id,
                anchor_polygon,
                anchor_mask,
                candidate_mask,
                intersected_area,
            )

    def build_page_flat_text_regions(
        self,
        page_image: Image,
        page_text_region_infos: Sequence[PageTextRegionInfo],
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

        text_region_flattener = TextRegionFlattener(
            typical_long_side_ratio_min=typical_long_side_ratio_min,
            text_region_polygon_dilate_ratio=text_region_polygon_dilate_ratio,
            image=page_image,
            text_region_polygons=text_region_polygons,
            grouped_char_polygons=grouped_char_polygons,
        )

        page_flat_text_regions: List[PageFlatTextRegion] = []
        for flattened_text_region in text_region_flattener.flattened_text_regions:
            assert flattened_text_region.flattened_char_polygons
            page_flat_text_regions.append(
                PageFlatTextRegion(
                    image=flattened_text_region.flattened_image,
                    char_polygons=flattened_text_region.flattened_char_polygons,
                )
            )
        return page_flat_text_regions

    @staticmethod
    def get_char_height(char_polygon: Polygon):
        assert len(char_polygon.points) == 4
        # Up left -> Down left.
        point0, point1, point2, point3 = char_polygon.points
        side0_length = math.hypot(point0.y - point3.y, point0.x - point3.x)
        side1_length = math.hypot(point1.y - point2.y, point1.x - point2.x)
        return (side0_length + side1_length) / 2

    def resize_page_flat_text_regions(
        self,
        page_flat_text_regions: Sequence[PageFlatTextRegion],
        rng: RandomGenerator,
    ):
        resized: List[PageFlatTextRegion] = []

        for page_flat_text_region in page_flat_text_regions:
            image = page_flat_text_region.image
            char_polygons = page_flat_text_region.char_polygons

            char_height_max = max(
                self.get_char_height(char_polygon)
                for char_polygon in page_flat_text_region.char_polygons
            )

            page_flat_text_region_resize_char_height = int(
                rng.integers(
                    self.config.page_flat_text_region_resize_char_height_min,
                    self.config.page_flat_text_region_resize_char_height_max + 1,
                )
            )
            scale = page_flat_text_region_resize_char_height / char_height_max

            resized_height = round(image.height * scale)
            resized_width = round(image.width * scale)

            resized.append(
                PageFlatTextRegion(
                    image=image.to_resized_image(
                        resized_height=resized_height,
                        resized_width=resized_width,
                    ),
                    char_polygons=[
                        char_polygon.to_conducted_resized_polygon(
                            image.shape,
                            resized_height=resized_height,
                            resized_width=resized_width,
                        ) for char_polygon in char_polygons
                    ],
                )
            )

        return resized

    def stack_page_flat_text_regions(
        self,
        page_flat_text_regions: Sequence[PageFlatTextRegion],
    ):
        pad = 2 * self.config.gap

        rect_packer = RectPacker(rotation=False)
        id_to_page_flat_text_region: Dict[int, PageFlatTextRegion] = {}

        # Add rectangle and bin.
        # NOTE: Only one bin is added, that is, packing all text region into one image.
        bin_width = 0
        bin_height = 0

        for pftr_id, page_flat_text_region in enumerate(page_flat_text_regions):
            id_to_page_flat_text_region[pftr_id] = page_flat_text_region
            rect_packer.add_rect(
                width=page_flat_text_region.width + pad,
                height=page_flat_text_region.height + pad,
                rid=pftr_id,
            )

            bin_width = max(bin_width, page_flat_text_region.width)
            bin_height += page_flat_text_region.height

        bin_width += pad
        bin_height += pad

        rect_packer.add_bin(width=bin_width, height=bin_height)
        rect_packer.pack()  # type: ignore

        boxes: List[Box] = []
        pftr_ids: List[int] = []
        for bin_idx, x, y, width, height, pftr_id in rect_packer.rect_list():
            assert bin_idx == 0
            boxes.append(Box(
                up=y,
                down=y + height - 1,
                left=x,
                right=x + width - 1,
            ))
            pftr_ids.append(pftr_id)

        page_height = max(box.down for box in boxes) + 1
        page_width = max(box.right for box in boxes) + 1

        image = Image.from_shape((page_height, page_width), value=0)
        char_polygons: List[Polygon] = []

        for box, pftr_id in zip(boxes, pftr_ids):
            page_flat_text_region = id_to_page_flat_text_region[pftr_id]
            assert page_flat_text_region.height + pad == box.height
            assert page_flat_text_region.width + pad == box.width

            up = box.up + self.config.gap
            left = box.left

            box = Box(
                up=up,
                down=up + page_flat_text_region.height - 1,
                left=left,
                right=left + page_flat_text_region.width - 1,
            )
            box.fill_image(image, page_flat_text_region.image)
            for char_polygon in page_flat_text_region.char_polygons:
                char_polygons.append(char_polygon.to_shifted_polygon(
                    offset_y=up,
                    offset_x=left,
                ))

        return image, char_polygons

    def run(self, input: PageTextRegionStepInput, rng: RandomGenerator):
        page_distortion_step_output = input.page_distortion_step_output
        page_image = page_distortion_step_output.page_image
        page_disconnected_text_region_collection = \
            page_distortion_step_output.page_disconnected_text_region_collection
        page_char_polygon_collection = page_distortion_step_output.page_char_polygon_collection

        page_resizing_step_output = input.page_resizing_step_output
        page_resized_text_line_mask = page_resizing_step_output.page_text_line_mask

        debug = None
        if self.config.debug:
            debug = DebugPageTextRegionStep()

        # Build R-tree to track text regions.
        # https://github.com/shapely/shapely/issues/640
        id_to_text_region_polygon: Dict[int, Polygon] = {}
        text_region_shapely_polygons: List[ShapelyPolygon] = []

        for polygon in page_disconnected_text_region_collection.to_polygons():
            shapely_polygon = polygon.to_shapely_polygon()
            id_to_text_region_polygon[id(shapely_polygon)] = polygon
            text_region_shapely_polygons.append(shapely_polygon)

        text_region_tree = STRtree(text_region_shapely_polygons)
        id_to_text_region_mask: Dict[int, Mask] = {}
        id_to_text_region_np_mask: Dict[int, np.ndarray] = {}

        # Get the precise text regions.
        precise_text_region_candidate_polygons: List[Polygon] = []
        for resized_polygon in page_resized_text_line_mask.to_disconnected_polygons():
            # Resize back to the shape after distortion.
            precise_polygon = resized_polygon.to_conducted_resized_polygon(
                page_resized_text_line_mask,
                resized_height=page_image.height,
                resized_width=page_image.width,
            )

            # Find all intersected text regions.
            for _, _, text_region_mask, precise_mask, _ in self.strtree_query_intersected_polygons(
                strtree=text_region_tree,
                id_to_anchor_polygon=id_to_text_region_polygon,
                id_to_anchor_mask=id_to_text_region_mask,
                id_to_anchor_np_mask=id_to_text_region_np_mask,
                candidate_polygon=precise_polygon,
            ):
                precise_text_region_candidate_polygons.extend(
                    self.generate_precise_text_region_candidate_polygons(
                        precise_mask=precise_mask,
                        text_region_mask=text_region_mask,
                    )
                )

        if debug:
            debug.page_image = page_image
            debug.precise_text_region_candidate_polygons = precise_text_region_candidate_polygons

        # Help gc.
        del id_to_text_region_polygon
        del text_region_shapely_polygons
        del text_region_tree
        del id_to_text_region_mask
        del id_to_text_region_np_mask

        # Bind char-level polygon to precise text region.
        id_to_precise_text_region_polygon: Dict[int, Polygon] = {}
        precise_text_region_shapely_polygons: List[ShapelyPolygon] = []

        for polygon in precise_text_region_candidate_polygons:
            shapely_polygon = polygon.to_shapely_polygon()
            id_to_precise_text_region_polygon[id(shapely_polygon)] = polygon
            precise_text_region_shapely_polygons.append(shapely_polygon)

        precise_text_region_tree = STRtree(precise_text_region_shapely_polygons)
        id_to_precise_text_region_mask: Dict[int, Mask] = {}
        id_to_precise_text_region_np_mask: Dict[int, np.ndarray] = {}

        id_to_char_polygons: DefaultDict[int, List[Polygon]] = defaultdict(list)
        for char_polygon in page_char_polygon_collection.polygons:
            best_precise_text_region_id = None
            max_intersected_area = 0

            for (
                precise_text_region_id,
                _,
                _,
                _,
                intersected_area,
            ) in self.strtree_query_intersected_polygons(
                strtree=precise_text_region_tree,
                id_to_anchor_polygon=id_to_precise_text_region_polygon,
                id_to_anchor_mask=id_to_precise_text_region_mask,
                id_to_anchor_np_mask=id_to_precise_text_region_np_mask,
                candidate_polygon=char_polygon,
            ):
                if intersected_area > max_intersected_area:
                    max_intersected_area = intersected_area
                    best_precise_text_region_id = precise_text_region_id

            if best_precise_text_region_id is not None:
                id_to_char_polygons[best_precise_text_region_id].append(char_polygon)

        page_text_region_infos: List[PageTextRegionInfo] = []
        for precise_text_region_shapely_polygon in precise_text_region_shapely_polygons:
            ptrsp_id = id(precise_text_region_shapely_polygon)
            if ptrsp_id not in id_to_char_polygons:
                # Not related to any char polygons.
                continue
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
        del id_to_precise_text_region_mask
        del id_to_precise_text_region_np_mask

        if debug:
            debug.page_text_region_infos = page_text_region_infos

        page_flat_text_regions = self.build_page_flat_text_regions(
            page_image=page_image,
            page_text_region_infos=page_text_region_infos,
            rng=rng,
        )

        if debug:
            debug.page_flat_text_regions = page_flat_text_regions

        # Resize text regions.
        page_flat_text_regions = self.resize_page_flat_text_regions(
            page_flat_text_regions,
            rng,
        )

        if debug:
            debug.resized_page_flat_text_regions = page_flat_text_regions

        # Stack text regions.
        image, char_polygons = self.stack_page_flat_text_regions(page_flat_text_regions)

        # Post uniform rotation.
        shape_before_rotate = image.shape
        rotate_angle = 0

        if self.config.enable_post_uniform_rotate:
            rotate_angle = int(rng.integers(0, 360))
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
