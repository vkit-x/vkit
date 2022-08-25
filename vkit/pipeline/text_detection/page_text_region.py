from typing import List, Optional, Dict, DefaultDict, Sequence
from collections import defaultdict
import math

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.errors import TopologicalError as ShapelyTopologicalError
from rectpack import newPacker as RectPacker

from vkit.element import Box, Polygon, Mask, Image
from vkit.engine.distortion.geometric.affine import rotate
from ..interface import PipelineStep, PipelineStepFactory
from .page_distortion import PageDistortionStepOutput
from .page_resizing import PageResizingStepOutput


@attrs.define
class PageTextRegionStepConfig:
    page_flat_text_region_resize_char_height_min: int = 32
    page_flat_text_region_resize_char_height_max: int = 36
    gap: int = 2
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
    debug: Optional[DebugPageTextRegionStep]


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
    def strtree_query_intersected_polygons(strtree: STRtree, shapely_polygon: ShapelyPolygon):
        for intersected_shapely_polygon in strtree.query(shapely_polygon):
            try:
                if not intersected_shapely_polygon.intersects(shapely_polygon):
                    continue
            except ShapelyTopologicalError:
                continue
            yield intersected_shapely_polygon

    @staticmethod
    def get_flat_rotation_angle(precise_text_region_polygon: Polygon):
        # Get minimum bounding rectangle.
        shapely_polygon = precise_text_region_polygon.to_shapely_polygon()
        minimum_rotated_rectangle = shapely_polygon.minimum_rotated_rectangle

        assert isinstance(minimum_rotated_rectangle, ShapelyPolygon)
        polygon = Polygon.from_shapely_polygon(minimum_rotated_rectangle)
        assert len(polygon.points) == 4

        # Get reference line.
        point0, point1, _, point3 = polygon.points
        side0_length = math.hypot(point0.y - point1.y, point0.x - point1.x)
        side1_length = math.hypot(point0.y - point3.y, point0.x - point3.x)

        point_a = point0
        if side0_length > side1_length:
            # Reference line (p0 -> p1).
            point_b = point1
        else:
            # Reference line (p0 -> p3).
            point_b = point3

        # Get angle of reference line in [0, 180]
        theta = np.arctan2(
            point_a.y - point_b.y,
            point_a.x - point_b.x,
        )
        theta = theta % np.pi
        angle = round(theta / np.pi * 180)

        # Get the angle for flattening.
        if angle <= 90:
            return 360 - angle
        else:
            return 180 - angle

    @staticmethod
    def build_page_flat_text_region(
        page_image: Image,
        page_text_region_info: PageTextRegionInfo,
    ):
        precise_text_region_polygon = page_text_region_info.precise_text_region_polygon
        char_polygons = page_text_region_info.char_polygons

        bounding_box = precise_text_region_polygon.to_bounding_box()
        # Need to make sure all char polygons are included.
        for char_polygon in char_polygons:
            char_bounding_box = char_polygon.to_bounding_box()
            bounding_box.up = min(bounding_box.up, char_bounding_box.up)
            bounding_box.down = max(bounding_box.down, char_bounding_box.down)
            bounding_box.left = min(bounding_box.left, char_bounding_box.left)
            bounding_box.right = max(bounding_box.right, char_bounding_box.right)

        shifted_precise_text_region_polygon = precise_text_region_polygon.to_shifted_polygon(
            y_offset=-bounding_box.up,
            x_offset=-bounding_box.left,
        )
        shifted_char_polygons = [
            char_polygon.to_shifted_polygon(
                y_offset=-bounding_box.up,
                x_offset=-bounding_box.left,
            ) for char_polygon in char_polygons
        ]

        # Build mask.
        mask = Mask.from_shapable(bounding_box)
        shifted_precise_text_region_polygon.fill_mask(mask)
        for shifted_char_polygon in shifted_char_polygons:
            shifted_char_polygon.fill_mask(mask)

        # Get image.
        image = bounding_box.extract_image(page_image)

        # Get the flat text region by rotation.
        angle = PageTextRegionStep.get_flat_rotation_angle(precise_text_region_polygon)

        # Rotate.
        rotated_result = rotate.distort(
            {'angle': angle},
            image=image,
            mask=mask,
            polygons=shifted_char_polygons,
        )
        rotated_image = rotated_result.image
        assert rotated_image
        rotated_mask = rotated_result.mask
        assert rotated_mask
        rotated_char_polygons = rotated_result.polygons
        assert rotated_char_polygons

        # Trim.
        np_hori_max = np.amax(rotated_mask.mat, axis=0)
        np_hori_nonzero = np.nonzero(np_hori_max)[0]
        assert len(np_hori_nonzero) >= 2
        left: int = np_hori_nonzero[0]
        right: int = np_hori_nonzero[-1]

        np_vert_max = np.amax(rotated_mask.mat, axis=1)
        np_vert_nonzero = np.nonzero(np_vert_max)[0]
        assert len(np_vert_nonzero) >= 2
        up: int = np_vert_nonzero[0]
        down: int = np_vert_nonzero[-1]

        rotated_image = rotated_image.to_cropped_image(
            up=up,
            down=down,
            left=left,
            right=right,
        )
        rotated_mask = rotated_mask.to_cropped_mask(
            up=up,
            down=down,
            left=left,
            right=right,
        )
        rotated_char_polygons = [
            rotated_char_polygon.to_shifted_polygon(
                y_offset=-up,
                x_offset=-left,
            ) for rotated_char_polygon in rotated_char_polygons
        ]

        # Hide inactive area.
        rotated_mask.to_inverted_mask().fill_image(rotated_image, value=0)

        return PageFlatTextRegion(
            image=rotated_image,
            char_polygons=rotated_char_polygons,
        )

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
                    y_offset=up,
                    x_offset=left,
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

        # Get the precise text regions.
        precise_text_region_candidate_polygons: List[Polygon] = []
        for resized_polygon in page_resized_text_line_mask.to_disconnected_polygons():
            # Resize back to the shape after distortion.
            precise_polygon = resized_polygon.to_conducted_resized_polygon(
                page_resized_text_line_mask,
                resized_height=page_image.height,
                resized_width=page_image.width,
            )

            # Prepare precise mask.
            precise_bounding_box = precise_polygon.to_bounding_box()
            precise_mask = Mask.from_shapable(precise_bounding_box)
            precise_mask = precise_mask.to_box_attached(precise_bounding_box)

            shifted_precise_polygon = precise_polygon.to_shifted_polygon(
                y_offset=-precise_bounding_box.up,
                x_offset=-precise_bounding_box.left,
            )
            shifted_precise_polygon.fill_mask(precise_mask)

            # Find all intersected text regions.
            id_to_text_region_mask: Dict[int, Mask] = {}
            precise_shapely_polygon = precise_polygon.to_shapely_polygon()

            for text_region_shapely_polygon in self.strtree_query_intersected_polygons(
                text_region_tree,
                precise_shapely_polygon,
            ):
                text_region_id = id(text_region_shapely_polygon)

                if text_region_id not in id_to_text_region_mask:
                    text_region_polygon = id_to_text_region_polygon[text_region_id]
                    text_region_bounding_box = text_region_polygon.to_bounding_box()
                    text_region_mask = Mask.from_shapable(text_region_bounding_box)
                    text_region_mask = text_region_mask.to_box_attached(text_region_bounding_box)

                    shifted_text_region_polygon = text_region_polygon.to_shifted_polygon(
                        y_offset=-text_region_bounding_box.up,
                        x_offset=-text_region_bounding_box.left,
                    )
                    shifted_text_region_polygon.fill_mask(text_region_mask)

                    id_to_text_region_mask[text_region_id] = text_region_mask

                precise_text_region_candidate_polygons.extend(
                    self.generate_precise_text_region_candidate_polygons(
                        precise_mask=precise_mask,
                        text_region_mask=id_to_text_region_mask[text_region_id],
                    )
                )

        if debug:
            debug.page_image = page_image
            debug.precise_text_region_candidate_polygons = precise_text_region_candidate_polygons

        # Help gc.
        del text_region_tree
        del text_region_shapely_polygons
        del id_to_text_region_polygon

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
            char_shapely_polygon = char_polygon.to_shapely_polygon()

            intersected: List[ShapelyPolygon] = []
            for precise_text_region_shapely_polygon in self.strtree_query_intersected_polygons(
                precise_text_region_tree,
                char_shapely_polygon,
            ):
                intersected.append(precise_text_region_shapely_polygon)  # type: ignore

            if not intersected:
                # Hit nothing.
                continue

            if len(intersected) == 1:
                # Simple case.
                id_to_char_polygons[id(intersected[0])].append(char_polygon)

            else:
                # If more than one, select the one with largest intersection area.
                largest_intersection_area = 0
                best_precise_text_region_shapely_polygon = None

                for precise_text_region_shapely_polygon in intersected:
                    intersection_result = \
                        precise_text_region_shapely_polygon.intersection(char_shapely_polygon)
                    intersection_area: int = intersection_result.area

                    if intersection_area > largest_intersection_area:
                        largest_intersection_area = intersection_area
                        best_precise_text_region_shapely_polygon = \
                            precise_text_region_shapely_polygon

                assert best_precise_text_region_shapely_polygon
                best_id = id(best_precise_text_region_shapely_polygon)
                id_to_char_polygons[best_id].append(char_polygon)

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
        del precise_text_region_tree
        del id_to_char_polygons
        del precise_text_region_shapely_polygons
        del id_to_precise_text_region_polygon

        if debug:
            debug.page_text_region_infos = page_text_region_infos

        # Generate flat text regions.
        page_flat_text_regions: List[PageFlatTextRegion] = []
        for page_text_region_info in page_text_region_infos:
            page_flat_text_regions.append(
                self.build_page_flat_text_region(
                    page_image=page_image,
                    page_text_region_info=page_text_region_info,
                )
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

        return PageTextRegionStepOutput(
            page_image=image,
            page_char_polygons=char_polygons,
            debug=debug,
        )


page_text_region_step_factory = PipelineStepFactory(PageTextRegionStep)
