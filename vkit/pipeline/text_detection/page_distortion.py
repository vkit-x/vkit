from typing import Optional, Union, Dict, Any, List, Sequence

import attrs
from numpy.random import RandomState
import numpy as np

from vkit.utility import PathType
from vkit.element import PointList, Point, Mask, ScoreMap, Image, Painter
from vkit.engine.distortion_policy.random_distortion import random_distortion_factory
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_text_line_label import PageTextLinePolygonCollection
from .page_assembler import PageAssemblerStep


@attrs.define
class PageDistortionStepConfig:
    random_distortion_factory_config: Optional[Union[Dict[str, Any], PathType]] = attrs.field(
        factory=lambda: {
            # NOTE: defocus blur and zoom in blur introduce labeling noise.
            # TODO: enhance those blurring methods for page.
            'disabled_policy_names': [
                'defocus_blur',
                'zoom_in_blur',
            ],
        }
    )
    debug_random_distortion: bool = False
    enable_distorted_text_line_mask: bool = True
    enable_distorted_text_line_height_score_map: bool = True
    enable_distorted_text_line_heights_debug: bool = False


@attrs.define
class PageDistortionStepOutput:
    page_image: Image
    page_text_line_polygon_collection: PageTextLinePolygonCollection
    page_random_distortion_debug_meta: Optional[Dict[str, Any]]
    page_text_line_mask: Optional[Mask]
    page_text_line_height_score_map: Optional[ScoreMap]
    page_text_line_heights: Optional[Sequence[float]]
    page_text_line_heights_debug_image: Optional[Image]


class PageDistortionStep(
    PipelineStep[
        PageDistortionStepConfig,
        PageDistortionStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageDistortionStepConfig):
        super().__init__(config)

        self.random_distortion = random_distortion_factory.create(
            self.config.random_distortion_factory_config
        )

    def run(self, state: PipelineState, rnd: RandomState):
        page_assembler_step_output = self.get_output(state, PageAssemblerStep)
        page = page_assembler_step_output.page
        page_text_line_polygon_collection = page.page_text_line_polygon_collection
        polygons = page_text_line_polygon_collection.polygons

        points = PointList()
        points.extend(page_text_line_polygon_collection.height_points_up)
        points.extend(page_text_line_polygon_collection.height_points_down)
        num_points = len(points)
        result = self.random_distortion.distort(
            image=page.image,
            polygons=polygons,
            points=points,
            rnd=rnd,
            debug=self.config.debug_random_distortion,
        )
        assert result.image
        assert result.polygons
        assert result.points

        # TODO: enhance interface.
        height_points_up = PointList(result.points[:num_points // 2])
        height_points_down = PointList(result.points[num_points // 2:])
        height_points_group_sizes = page_text_line_polygon_collection.height_points_group_sizes
        assert len(result.polygons) == len(height_points_group_sizes)
        page_distorted_text_line_polygon_collection = PageTextLinePolygonCollection(
            height=result.image.height,
            width=result.image.width,
            polygons=result.polygons,
            height_points_group_sizes=height_points_group_sizes,
            height_points_up=height_points_up,
            height_points_down=height_points_down,
        )

        mask: Optional[Mask] = None
        if self.config.enable_distorted_text_line_mask:
            mask = Mask.from_shapable(result.image)
            for polygon in result.polygons:
                polygon.fill_mask(mask)

        score_map: Optional[ScoreMap] = None
        page_distorted_text_line_heights: Optional[List[float]] = None
        page_text_line_heights_debug_image: Optional[Image] = None

        if self.config.enable_distorted_text_line_height_score_map:
            np_height_points_up = height_points_up.to_np_array()
            np_height_points_down = height_points_down.to_np_array()
            np_heights: np.ndarray = np.linalg.norm(
                np_height_points_down - np_height_points_up,
                axis=1,
            )
            # Add one to compensate.
            np_heights += 1
            assert sum(height_points_group_sizes) == np_heights.shape[0]

            page_distorted_text_line_heights = []
            score_map = ScoreMap.from_shapable(result.image, score_as_prob=False)
            begin = 0
            for polygon, group_size in zip(result.polygons, height_points_group_sizes):
                end = begin + group_size - 1
                page_distorted_text_line_height = float(np_heights[begin:end + 1].mean())
                page_distorted_text_line_heights.append(page_distorted_text_line_height)
                polygon.fill_score_map(
                    score_map=score_map,
                    value=page_distorted_text_line_height,
                )
                begin = end + 1

            if self.config.enable_distorted_text_line_heights_debug:
                painter = Painter.create(result.image)
                painter.paint_polygons(result.polygons)

                texts: List[str] = []
                points: List[Point] = []
                for polygon, height in zip(result.polygons, page_distorted_text_line_heights):
                    texts.append(f'{height:.1f}')
                    points.append(polygon.get_center_point())
                painter.paint_texts(texts, points, alpha=1.0)

                page_text_line_heights_debug_image = painter.image

        return PageDistortionStepOutput(
            page_image=result.image,
            page_text_line_polygon_collection=page_distorted_text_line_polygon_collection,
            page_random_distortion_debug_meta=result.meta,
            page_text_line_mask=mask,
            page_text_line_height_score_map=score_map,
            page_text_line_heights=page_distorted_text_line_heights,
            page_text_line_heights_debug_image=page_text_line_heights_debug_image,
        )


page_distortion_step_factory = PipelineStepFactory(PageDistortionStep)
