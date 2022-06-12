from typing import List, Sequence, Optional

import attrs
from numpy.random import RandomState

from vkit.element import Point, PointList, Box, Mask, ScoreMap, Polygon
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_text_line import PageTextLineStep, PageTextLineCollection


@attrs.define
class PageTextLinePolygonCollection:
    height: int
    width: int
    polygons: Sequence[Polygon]
    height_points_group_sizes: Sequence[int]
    height_points_up: PointList
    height_points_down: PointList


@attrs.define
class PageTextLineLabelStepConfig:
    num_sample_height_points: int = 3
    enable_text_line_mask: bool = False
    enable_boundary_mask: bool = False
    boundary_dilate_ratio: float = 0.5
    enable_boundary_score_map: bool = False


@attrs.define
class PageTextLineLabelStepOutput:
    page_text_line_polygon_collection: PageTextLinePolygonCollection
    page_text_line_mask: Optional[Mask]
    page_text_line_boundary_mask: Optional[Mask]
    page_text_line_and_boundary_mask: Optional[Mask]
    page_text_line_boundary_score_map: Optional[ScoreMap]


class PageTextLineLabelStep(
    PipelineStep[
        PageTextLineLabelStepConfig,
        PageTextLineLabelStepOutput,
    ]
):  # yapf: disable

    def generate_page_text_line_polygon_collection(
        self,
        page_text_line_collection: PageTextLineCollection,
    ):
        text_line_polygons: List[Polygon] = []

        height_points_group_sizes: List[int] = []
        height_points_up = PointList()
        height_points_down = PointList()

        text_lines = page_text_line_collection.text_lines
        for text_line in text_lines:
            text_line_polygons.append(text_line.to_polygon())

            cur_height_points_up = text_line.get_height_points(
                num_points=self.config.num_sample_height_points,
                is_up=True,
            )
            cur_height_points_down = text_line.get_height_points(
                num_points=self.config.num_sample_height_points,
                is_up=False,
            )
            height_points_group_size = len(cur_height_points_up)
            assert height_points_group_size == len(cur_height_points_down)
            assert height_points_group_size > 0
            height_points_group_sizes.append(height_points_group_size)
            height_points_up.extend(cur_height_points_up)
            height_points_down.extend(cur_height_points_down)

        return PageTextLinePolygonCollection(
            height=page_text_line_collection.height,
            width=page_text_line_collection.width,
            polygons=text_line_polygons,
            height_points_group_sizes=height_points_group_sizes,
            height_points_up=height_points_up,
            height_points_down=height_points_down,
        )

    def generate_page_text_line_mask(self, page_text_line_collection: PageTextLineCollection):
        page_text_line_mask = Mask.from_shape(page_text_line_collection.shape)

        text_lines = page_text_line_collection.text_lines
        for text_line in text_lines:
            text_line.box.fill_mask(page_text_line_mask)
        return page_text_line_mask

    def generate_text_line_boxes_and_dilated_boxes(
        self, page_text_line_collection: PageTextLineCollection
    ):
        text_lines = page_text_line_collection.text_lines
        text_lines = sorted(text_lines, key=lambda tl: tl.font_size, reverse=True)

        boxes: List[Box] = []
        dilated_boxes: List[Box] = []

        for text_line in text_lines:
            box = text_line.box
            boxes.append(box)

            dilated_box = box.to_dilated_box(self.config.boundary_dilate_ratio, clip_long_side=True)
            dilated_box = dilated_box.to_clipped_box(page_text_line_collection.shape)
            dilated_boxes.append(dilated_box)

        return boxes, dilated_boxes

    @staticmethod
    def generate_dilated_only_boxes(
        box: Box,
        dilated_box: Box,
    ):
        dilated_up_box = attrs.evolve(
            dilated_box,
            down=box.up - 1,
        )
        if dilated_up_box.up > dilated_box.down:
            dilated_up_box = None

        dilated_down_box = attrs.evolve(
            dilated_box,
            up=box.down + 1,
        )
        if dilated_down_box.up > dilated_down_box.down:
            dilated_down_box = None

        dilated_left_box = attrs.evolve(
            box,
            left=dilated_box.left,
            right=box.left - 1,
        )
        if dilated_left_box.left > dilated_left_box.right:
            dilated_left_box = None

        dilated_right_box = attrs.evolve(
            box,
            left=box.right + 1,
            right=dilated_box.right,
        )
        if dilated_right_box.left > dilated_right_box.right:
            dilated_right_box = None

        return (
            dilated_up_box,
            dilated_down_box,
            dilated_left_box,
            dilated_right_box,
        )

    def generate_page_text_line_boundary_masks(
        self,
        page_text_line_collection: PageTextLineCollection,
        boxes: Sequence[Box],
        dilated_boxes: Sequence[Box],
        page_text_line_mask: Mask,
    ):
        boundary_mask = Mask.from_shape(page_text_line_collection.shape)

        for box, dilated_box in zip(boxes, dilated_boxes):
            dilated_only_boxes = PageTextLineLabelStep.generate_dilated_only_boxes(box, dilated_box)
            for dilated_only_box in dilated_only_boxes:
                if dilated_only_box:
                    dilated_only_box.fill_mask(boundary_mask)

        page_text_line_mask.fill_mask(boundary_mask, 0)

        text_line_and_boundary_mask = boundary_mask.copy()
        page_text_line_mask.fill_mask(text_line_and_boundary_mask)

        return boundary_mask, text_line_and_boundary_mask

    def generate_page_text_line_boundary_score_map(
        self,
        page_text_line_collection: PageTextLineCollection,
        boxes: Sequence[Box],
        dilated_boxes: Sequence[Box],
        page_text_line_boundary_mask: Mask,
    ):
        boundary_score_map = ScoreMap.from_shape(page_text_line_collection.shape, value=1.0)

        for box, dilated_box in zip(boxes, dilated_boxes):
            (
                dilated_up_box,
                dilated_down_box,
                dilated_left_box,
                dilated_right_box,
            ) = PageTextLineLabelStep.generate_dilated_only_boxes(box, dilated_box)

            if dilated_up_box:
                boundary_score_map.fill_by_quad_interpolation(
                    point0=Point(y=box.up, x=box.right),
                    point1=Point(y=box.up, x=box.left),
                    point2=Point(y=dilated_box.up, x=dilated_box.left),
                    point3=Point(y=dilated_box.up, x=dilated_box.right),
                    func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1],
                    keep_min_value=True,
                )

            if dilated_down_box:
                boundary_score_map.fill_by_quad_interpolation(
                    point0=Point(y=box.down, x=box.left),
                    point1=Point(y=box.down, x=box.right),
                    point2=Point(y=dilated_box.down, x=dilated_box.right),
                    point3=Point(y=dilated_box.down, x=dilated_box.left),
                    func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1],
                    keep_min_value=True,
                )

            if dilated_left_box:
                boundary_score_map.fill_by_quad_interpolation(
                    point0=Point(y=box.up, x=box.left),
                    point1=Point(y=box.down, x=box.left),
                    point2=Point(y=dilated_box.down, x=dilated_box.left),
                    point3=Point(y=dilated_box.up, x=dilated_box.left),
                    func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1],
                    keep_min_value=True,
                )

            if dilated_right_box:
                boundary_score_map.fill_by_quad_interpolation(
                    point0=Point(y=box.down, x=box.right),
                    point1=Point(y=box.up, x=box.right),
                    point2=Point(y=dilated_box.up, x=dilated_box.right),
                    point3=Point(y=dilated_box.down, x=dilated_box.right),
                    func_np_uv_to_mat=lambda np_uv: np_uv[:, :, 1],
                    keep_min_value=True,
                )

        page_text_line_boundary_mask.to_inverted_mask().fill_score_map(boundary_score_map, 0.0)
        return boundary_score_map

    def run(self, state: PipelineState, rnd: RandomState):
        page_text_line_step_output = self.get_output(state, PageTextLineStep)
        page_text_line_collection = page_text_line_step_output.page_text_line_collection

        page_text_line_polygon_collection = self.generate_page_text_line_polygon_collection(
            page_text_line_collection,
        )

        page_text_line_mask: Optional[Mask] = None
        page_text_line_boundary_mask: Optional[Mask] = None
        page_text_line_and_boundary_mask: Optional[Mask] = None
        page_text_line_boundary_score_map: Optional[ScoreMap] = None

        if self.config.enable_text_line_mask:
            page_text_line_mask = self.generate_page_text_line_mask(page_text_line_collection)

            boxes, dilated_boxes = self.generate_text_line_boxes_and_dilated_boxes(
                page_text_line_collection
            )
            if self.config.enable_boundary_mask:
                (
                    page_text_line_boundary_mask,
                    page_text_line_and_boundary_mask,
                ) = self.generate_page_text_line_boundary_masks(
                    page_text_line_collection,
                    boxes,
                    dilated_boxes,
                    page_text_line_mask,
                )

                if self.config.enable_boundary_score_map:
                    page_text_line_boundary_score_map = \
                        self.generate_page_text_line_boundary_score_map(
                            page_text_line_collection,
                            boxes,
                            dilated_boxes,
                            page_text_line_boundary_mask,
                        )

        return PageTextLineLabelStepOutput(
            page_text_line_polygon_collection=page_text_line_polygon_collection,
            page_text_line_mask=page_text_line_mask,
            page_text_line_boundary_mask=page_text_line_boundary_mask,
            page_text_line_and_boundary_mask=page_text_line_and_boundary_mask,
            page_text_line_boundary_score_map=page_text_line_boundary_score_map,
        )


page_text_line_label_step_factory = PipelineStepFactory(PageTextLineLabelStep)
