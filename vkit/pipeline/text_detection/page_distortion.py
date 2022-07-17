from typing import Optional, Union, Mapping, Any, List, Tuple, Sequence

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np

from vkit.utility import PathType
from vkit.element import PointList, Point, Polygon, Mask, ScoreMap, Image, Painter
from vkit.engine.distortion_policy.random_distortion import random_distortion_factory
from ..interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineState,
)
from .page_text_line_label import PageCharPolygonCollection, PageTextLinePolygonCollection
from .page_assembler import PageAssemblerStep


@attrs.define
class PageDistortionStepConfig:
    random_distortion_factory_config: Optional[Union[Mapping[str, Any], PathType]] = attrs.field(
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
    enable_distorted_char_mask: bool = True
    enable_distorted_char_height_score_map: bool = True
    enable_distorted_char_heights_debug: bool = False
    enable_distorted_text_line_mask: bool = True
    enable_distorted_text_line_height_score_map: bool = True
    enable_distorted_text_line_heights_debug: bool = False


@attrs.define
class PageDistortionStepOutput:
    page_image: Image
    page_random_distortion_debug_meta: Optional[Mapping[str, Any]]
    page_char_polygon_collection: PageCharPolygonCollection
    page_char_mask: Optional[Mask]
    page_char_height_score_map: Optional[ScoreMap]
    page_char_heights: Optional[Sequence[float]]
    page_char_heights_debug_image: Optional[Image]
    page_text_line_polygon_collection: PageTextLinePolygonCollection
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

    def generate_text_line_labelings(
        self,
        distorted_image: Image,
        text_line_polygons: Sequence[Polygon],
        text_line_height_points_up: PointList,
        text_line_height_points_down: PointList,
        text_line_height_points_group_sizes: Sequence[int],
    ):
        text_line_mask: Optional[Mask] = None
        if self.config.enable_distorted_text_line_mask:
            text_line_mask = Mask.from_shapable(distorted_image)
            for polygon in text_line_polygons:
                polygon.fill_mask(text_line_mask)

        text_line_height_score_map: Optional[ScoreMap] = None
        text_line_heights: Optional[List[float]] = None
        text_line_heights_debug_image: Optional[Image] = None

        if self.config.enable_distorted_text_line_height_score_map:
            np_height_points_up = text_line_height_points_up.to_np_array()
            np_height_points_down = text_line_height_points_down.to_np_array()
            np_heights: np.ndarray = np.linalg.norm(
                np_height_points_down - np_height_points_up,
                axis=1,
            )
            # Add one to compensate.
            np_heights += 1
            assert sum(text_line_height_points_group_sizes) == np_heights.shape[0]

            text_line_heights = []
            text_line_height_score_map = ScoreMap.from_shapable(distorted_image, is_prob=False)
            begin = 0
            for polygon, group_size in zip(text_line_polygons, text_line_height_points_group_sizes):
                end = begin + group_size - 1
                text_line_height = float(np_heights[begin:end + 1].mean())
                text_line_heights.append(text_line_height)
                polygon.fill_score_map(
                    score_map=text_line_height_score_map,
                    value=text_line_height,
                )
                begin = end + 1

            if self.config.enable_distorted_text_line_heights_debug:
                painter = Painter.create(distorted_image)
                painter.paint_polygons(text_line_polygons)

                texts: List[str] = []
                points: List[Point] = []
                for polygon, height in zip(text_line_polygons, text_line_heights):
                    texts.append(f'{height:.1f}')
                    points.append(polygon.get_center_point())
                painter.paint_texts(texts, points, alpha=1.0)

                text_line_heights_debug_image = painter.image

        return (
            text_line_mask,
            text_line_height_score_map,
            text_line_heights,
            text_line_heights_debug_image,
        )

    def generate_char_labelings(
        self,
        distorted_image: Image,
        char_polygons: Sequence[Polygon],
        char_height_points_up: PointList,
        char_height_points_down: PointList,
    ):
        char_mask: Optional[Mask] = None
        if self.config.enable_distorted_text_line_mask:
            char_mask = Mask.from_shapable(distorted_image)
            for polygon in char_polygons:
                polygon.fill_mask(char_mask)

        char_height_score_map: Optional[ScoreMap] = None
        char_heights: Optional[List[float]] = None
        char_heights_debug_image: Optional[Image] = None

        if self.config.enable_distorted_char_height_score_map:
            np_height_points_up = char_height_points_up.to_np_array()
            np_height_points_down = char_height_points_down.to_np_array()
            np_heights: np.ndarray = np.linalg.norm(
                np_height_points_down - np_height_points_up,
                axis=1,
            )
            # Add one to compensate.
            np_heights += 1

            # Fill from large height to small height,
            # in order to preserve small height labeling when two char boxes overlapped.
            sorted_char_polygon_indices: Tuple[int, ...] = tuple(reversed(np_heights.argsort()))

            char_heights = [0.0] * len(char_polygons)
            char_height_score_map = ScoreMap.from_shapable(distorted_image, is_prob=False)

            for idx in sorted_char_polygon_indices:
                polygon = char_polygons[idx]
                char_height = float(np_heights[idx])
                char_heights[idx] = char_height
                polygon.fill_score_map(
                    score_map=char_height_score_map,
                    value=char_height,
                )

            if self.config.enable_distorted_char_heights_debug:
                painter = Painter.create(distorted_image)
                painter.paint_polygons(char_polygons)

                texts: List[str] = []
                points: List[Point] = []
                for polygon, height in zip(char_polygons, char_heights):
                    texts.append(f'{height:.1f}')
                    points.append(polygon.get_center_point())
                painter.paint_texts(texts, points, alpha=1.0)

                char_heights_debug_image = painter.image

        return (
            char_mask,
            char_height_score_map,
            char_heights,
            char_heights_debug_image,
        )

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_assembler_step_output = state.get_pipeline_step_output(PageAssemblerStep)
        page = page_assembler_step_output.page
        page_char_polygon_collection = page.page_char_polygon_collection
        page_text_line_polygon_collection = page.page_text_line_polygon_collection

        polygons: List[Polygon] = []
        points = PointList()

        # Char level.
        polygons.extend(page_char_polygon_collection.polygons)
        points.extend(page_char_polygon_collection.height_points_up)
        points.extend(page_char_polygon_collection.height_points_down)

        # Text line level.
        polygons.extend(page_text_line_polygon_collection.polygons)
        points.extend(page_text_line_polygon_collection.height_points_up)
        points.extend(page_text_line_polygon_collection.height_points_down)

        # Distort.
        result = self.random_distortion.distort(
            image=page.image,
            polygons=polygons,
            points=points,
            rng=rng,
            debug=self.config.debug_random_distortion,
        )
        assert result.image
        assert result.polygons
        assert result.points

        # Unpack.
        # TODO: enhance interface for packing & unpacking.
        num_chars = len(page_char_polygon_collection.polygons)
        text_line_points_begin = 2 * num_chars

        char_polygons = result.polygons[:num_chars]

        char_points = result.points[:text_line_points_begin]
        char_height_points_up = PointList(char_points[:num_chars])
        char_height_points_down = PointList(char_points[num_chars:])

        text_line_polygons = result.polygons[num_chars:]

        text_line_points = result.points[text_line_points_begin:]
        num_half_text_line_points = len(text_line_points) // 2
        text_line_height_points_up = PointList(text_line_points[:num_half_text_line_points])
        text_line_height_points_down = PointList(text_line_points[num_half_text_line_points:])
        text_line_height_points_group_sizes = \
            page_text_line_polygon_collection.height_points_group_sizes
        assert len(text_line_polygons) == len(text_line_height_points_group_sizes)
        assert len(text_line_height_points_up) == len(text_line_height_points_down)

        # Labelings.
        (
            text_line_mask,
            text_line_height_score_map,
            text_line_heights,
            text_line_heights_debug_image,
        ) = self.generate_text_line_labelings(
            distorted_image=result.image,
            text_line_polygons=text_line_polygons,
            text_line_height_points_up=text_line_height_points_up,
            text_line_height_points_down=text_line_height_points_down,
            text_line_height_points_group_sizes=text_line_height_points_group_sizes,
        )
        (
            char_mask,
            char_height_score_map,
            char_heights,
            char_heights_debug_image,
        ) = self.generate_char_labelings(
            distorted_image=result.image,
            char_polygons=char_polygons,
            char_height_points_up=char_height_points_up,
            char_height_points_down=char_height_points_down,
        )

        return PageDistortionStepOutput(
            page_image=result.image,
            page_random_distortion_debug_meta=result.meta,
            page_char_polygon_collection=PageCharPolygonCollection(
                height=result.image.height,
                width=result.image.width,
                polygons=char_polygons,
                height_points_up=char_height_points_up,
                height_points_down=char_height_points_down,
            ),
            page_char_mask=char_mask,
            page_char_height_score_map=char_height_score_map,
            page_char_heights=char_heights,
            page_char_heights_debug_image=char_heights_debug_image,
            page_text_line_polygon_collection=PageTextLinePolygonCollection(
                height=result.image.height,
                width=result.image.width,
                polygons=text_line_polygons,
                height_points_group_sizes=text_line_height_points_group_sizes,
                height_points_up=text_line_height_points_up,
                height_points_down=text_line_height_points_down,
            ),
            page_text_line_mask=text_line_mask,
            page_text_line_height_score_map=text_line_height_score_map,
            page_text_line_heights=text_line_heights,
            page_text_line_heights_debug_image=text_line_heights_debug_image,
        )


page_distortion_step_factory = PipelineStepFactory(PageDistortionStep)
