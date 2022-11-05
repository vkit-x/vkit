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
from typing import Optional, Union, Mapping, Any, List, Tuple, Sequence, TypeVar, Generic
import itertools
from enum import unique, Enum
import math

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np

from vkit.utility import PathType
from vkit.element import (
    Point,
    PointList,
    Box,
    Polygon,
    Mask,
    ScoreMap,
    Image,
)
from vkit.mechanism.distortion_policy import (
    random_distortion_factory,
    RandomDistortionDebug,
)
from vkit.mechanism.painter import Painter
from vkit.engine.char_heatmap.default import build_np_distance
from ..interface import PipelineStep, PipelineStepFactory
from .page_layout import DisconnectedTextRegion, NonTextRegion
from .page_text_line_label import (
    PageCharPolygonCollection,
    PageTextLinePolygonCollection,
)
from .page_assembler import (
    PageAssemblerStepOutput,
    PageDisconnectedTextRegionCollection,
    PageNonTextRegionCollection,
)


@unique
class CharMaskStyle(Enum):
    DEFAULT = 'default'
    EXTERNAL_ELLIPSE = 'external_ellipse'


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
    enable_debug_random_distortion: bool = False
    enable_distorted_char_mask: bool = True
    char_mask_style: CharMaskStyle = CharMaskStyle.DEFAULT
    char_mask_style_external_ellipse_internal_side_length: int = 40
    enable_distorted_char_height_score_map: bool = True
    enable_debug_distorted_char_heights: bool = False
    enable_distorted_text_line_mask: bool = True
    enable_distorted_text_line_height_score_map: bool = True
    enable_debug_distorted_text_line_heights: bool = False


@attrs.define
class PageDistortionStepInput:
    page_assembler_step_output: PageAssemblerStepOutput


@attrs.define
class PageDistortionStepOutput:
    page_image: Image
    page_random_distortion_debug: Optional[RandomDistortionDebug]
    page_active_mask: Mask
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
    page_disconnected_text_region_collection: PageDisconnectedTextRegionCollection
    page_non_text_region_collection: PageNonTextRegionCollection


_E = TypeVar('_E', Point, Polygon)


class ElementFlattener(Generic[_E]):

    def __init__(self, grouped_elements: Sequence[Sequence[_E]]):
        self.grouped_elements = grouped_elements
        self.group_sizes = [len(elements) for elements in grouped_elements]

    def flatten(self):
        return tuple(itertools.chain.from_iterable(self.grouped_elements))

    def unflatten(self, flattened_elements: Sequence[_E]) -> Sequence[Sequence[_E]]:
        assert len(flattened_elements) == sum(self.group_sizes)
        grouped_elements: List[Sequence[_E]] = []
        begin = 0
        for group_size in self.group_sizes:
            end = begin + group_size
            grouped_elements.append(flattened_elements[begin:end])
            begin = end
        return grouped_elements


class PageDistortionStep(
    PipelineStep[
        PageDistortionStepConfig,
        PageDistortionStepInput,
        PageDistortionStepOutput,
    ]
):  # yapf: disable

    def __init__(self, config: PageDistortionStepConfig):
        super().__init__(config)

        self.random_distortion = random_distortion_factory.create(
            self.config.random_distortion_factory_config
        )

    @classmethod
    def fill_page_inactive_region(
        cls,
        page_image: Image,
        page_active_mask: Mask,
        page_bottom_layer_image: Image,
    ):
        assert page_image.shape == page_active_mask.shape

        if page_bottom_layer_image.shape != page_image.shape:
            page_bottom_layer_image = page_bottom_layer_image.to_resized_image(
                resized_height=page_image.height,
                resized_width=page_image.width,
            )

        page_active_mask.to_inverted_mask().fill_image(page_image, page_bottom_layer_image)

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
            np_height_points_up = text_line_height_points_up.to_smooth_np_array()
            np_height_points_down = text_line_height_points_down.to_smooth_np_array()
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

            if self.config.enable_debug_distorted_text_line_heights:
                painter = Painter.create(distorted_image)
                painter.paint_polygons(text_line_polygons)

                texts: List[str] = []
                points = PointList()
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

    def generate_char_mask(self, shape: Tuple[int, int], char_polygons: Sequence[Polygon]):
        char_mask = Mask.from_shape(shape)

        if self.config.char_mask_style == CharMaskStyle.DEFAULT:
            for polygon in char_polygons:
                polygon.fill_mask(char_mask)

        elif self.config.char_mask_style == CharMaskStyle.EXTERNAL_ELLIPSE:
            internal_side_length = self.config.char_mask_style_external_ellipse_internal_side_length
            external_radius = math.ceil(internal_side_length / math.sqrt(2))

            # Build distances to the center point.
            np_distance = build_np_distance(external_radius)

            # Build mask.
            external_circle_mask = Mask(mat=(np_distance <= external_radius).astype(np.uint8))
            external_circle_ratio = external_circle_mask.height / internal_side_length

            # TODO: NOOOOOOOO, it's after distortion!.
            # # Fill mask.
            # for polygon in char_polygons:
            #     # Recover to box.
            #     assert polygon.num_points == 4
            #     (
            #         up_left,
            #         up_right,
            #         down_right,
            #         down_left,
            #     ) = polygon.points
            #     assert up_left.y == up_right.y
            #     up = up_left.y
            #     assert down_left.y == down_right.y
            #     down = down_left.y
            #     assert up_left.x == down_left.x
            #     left = up_left.x
            #     assert up_right.x == down_right.x
            #     right = up_right.x

            #     height = down + 1 - up
            #     width = right + 1 - left

            #     # Resize to build mask for external ellipse pattern.
            #     resized_height = round(external_circle_ratio * height)
            #     resized_width = round(external_circle_ratio * width)

            #     external_ellipse_mask = external_circle_mask.to_resized_mask(
            #         resized_height=resized_height,
            #         resized_width=resized_width,
            #     )

            #     # Placement.
            #     # TODO: Refactor.
            #     pad_up = (resized_height - height) // 2
            #     target_up = up - pad_up
            #     if target_up < 0:
            #         pad_up += target_up
            #         assert pad_up >= 0
            #         target_up = 0

            #     pad_down = (resized_height - height) // 2
            #     target_down = down + pad_down
            #     if target_down >= char_mask.height:
            #         pad_down -= (target_down + 1 - char_mask.height)
            #         assert pad_down >= 0
            #         target_down = char_mask.height - 1

            #     pad_left = (resized_width - width) // 2
            #     target_left = left - pad_left
            #     if target_left < 0:
            #         pad_left += target_left
            #         assert pad_left >= 0
            #         target_left = 0

            #     pad_right = (resized_width - width) // 2
            #     target_right = right + pad_right
            #     if target_right >= char_mask.width:
            #         pad_right -= (target_right + 1 - char_mask.width)
            #         assert pad_right >= 0
            #         target_right = char_mask.width - 1

            #     target_box = Box(
            #         up=target_up,
            #         down=target_down,
            #         left=target_left,
            #         right=target_right,
            #     )
            #     trimmed_box = Box(
            #         up=pad_up,
            #         down=pad_down + height + pad_down - 1,
            #         left=pad_left,
            #         right=pad_left + width + pad_right - 1,
            #     )
            #     assert target_box.shape == trimmed_box.shape
            #     target_box.fill_mask(
            #         char_mask,
            #         trimmed_box.extract_mask(external_ellipse_mask),
            #         keep_max_value=True,
            #     )

        else:
            raise NotImplementedError()

        return char_mask

    def generate_char_labelings(
        self,
        distorted_image: Image,
        char_polygons: Sequence[Polygon],
        char_height_points_up: PointList,
        char_height_points_down: PointList,
    ):
        char_mask: Optional[Mask] = None
        if self.config.enable_distorted_char_mask:
            char_mask = self.generate_char_mask(distorted_image.shape, char_polygons)

        char_height_score_map: Optional[ScoreMap] = None
        char_heights: Optional[List[float]] = None
        char_heights_debug_image: Optional[Image] = None

        if self.config.enable_distorted_char_height_score_map:
            np_height_points_up = char_height_points_up.to_smooth_np_array()
            np_height_points_down = char_height_points_down.to_smooth_np_array()
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

            if self.config.enable_debug_distorted_char_heights:
                painter = Painter.create(distorted_image)
                painter.paint_polygons(char_polygons)

                texts: List[str] = []
                points = PointList()
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

    def run(self, input: PageDistortionStepInput, rng: RandomGenerator):
        page_assembler_step_output = input.page_assembler_step_output
        page = page_assembler_step_output.page
        page_bottom_layer_image = page.page_bottom_layer_image
        page_char_polygon_collection = page.page_char_polygon_collection
        page_text_line_polygon_collection = page.page_text_line_polygon_collection
        page_disconnected_text_region_collection = page.page_disconnected_text_region_collection
        page_non_text_region_collection = page.page_non_text_region_collection

        # Flatten.
        polygon_flattener = ElementFlattener([
            # Char level.
            page_char_polygon_collection.polygons,
            # Text line level.
            page_text_line_polygon_collection.polygons,
            # For char-level polygon regression.
            tuple(page_disconnected_text_region_collection.to_polygons()),
            # For sampling negative text region area.
            tuple(page_non_text_region_collection.to_polygons()),
        ])
        point_flattener = ElementFlattener([
            # Char level.
            page_char_polygon_collection.height_points_up,
            page_char_polygon_collection.height_points_down,
            # Text line level.
            page_text_line_polygon_collection.height_points_up,
            page_text_line_polygon_collection.height_points_down,
        ])

        # Distort.
        page_random_distortion_debug = None
        if self.config.enable_debug_random_distortion:
            page_random_distortion_debug = RandomDistortionDebug()

        page_active_mask = Mask.from_shapable(page.image, value=1)
        # To mitigate a bug in cv.remap, in which the border interpolation is wrong.
        # This mitigation DO remove 1-pixel width border, but it should be fine.
        with page_active_mask.writable_context:
            page_active_mask.mat[0] = 0
            page_active_mask.mat[-1] = 0
            page_active_mask.mat[:, 0] = 0
            page_active_mask.mat[:, -1] = 0

        result = self.random_distortion.distort(
            image=page.image,
            mask=page_active_mask,
            polygons=polygon_flattener.flatten(),
            points=PointList(point_flattener.flatten()),
            rng=rng,
            debug=page_random_distortion_debug,
        )
        assert result.image
        assert result.mask
        assert result.polygons
        assert result.points

        # Fill inplace the inactive (black) region with page_bottom_layer_image.
        self.fill_page_inactive_region(
            page_image=result.image,
            page_active_mask=result.mask,
            page_bottom_layer_image=page_bottom_layer_image,
        )

        # Unflatten.
        (
            # Char level.
            char_polygons,
            # Text line level.
            text_line_polygons,
            # For char-level polygon regression.
            disconnected_text_region_polygons,
            # For sampling negative text region area.
            non_text_region_polygons,
        ) = polygon_flattener.unflatten(result.polygons)

        (
            # Char level.
            char_height_points_up,
            char_height_points_down,
            # Text line level.
            text_line_height_points_up,
            text_line_height_points_down,
        ) = map(PointList, point_flattener.unflatten(result.points))

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
            page_random_distortion_debug=page_random_distortion_debug,
            page_active_mask=result.mask,
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
            page_disconnected_text_region_collection=PageDisconnectedTextRegionCollection(
                disconnected_text_regions=[
                    DisconnectedTextRegion(disconnected_text_region_polygon)
                    for disconnected_text_region_polygon in disconnected_text_region_polygons
                ],
            ),
            page_non_text_region_collection=PageNonTextRegionCollection(
                non_text_regions=[
                    NonTextRegion(non_text_region_polygon)
                    for non_text_region_polygon in non_text_region_polygons
                ],
            )
        )


page_distortion_step_factory = PipelineStepFactory(PageDistortionStep)
