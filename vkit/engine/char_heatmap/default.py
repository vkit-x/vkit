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
from typing import Optional

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv

from vkit.element import Mask, ScoreMap, ElementSetOperationMode
from vkit.engine.interface import (
    Engine,
    EngineExecutorFactory,
    NoneTypeEngineInitResource,
)
from .type import CharHeatmapEngineRunConfig, CharHeatmap


@attrs.define
class CharHeatmapDefaultEngineInitConfig:
    # Adjust the std. The std gets smaller as the distance factor gets larger.
    # The activated area shrinks to the center as the std gets smaller.
    # https://colab.research.google.com/drive/1TQ1-BTisMYZHIRVVNpVwDFPviXYMhT7A
    gaussian_map_distance_factor: float = 2.25
    gaussian_map_char_radius: int = 25
    gaussian_map_preserving_score_min: float = 0.8
    weight_neutralized_score_map: float = 0.4


@attrs.define
class CharHeatmapDefaultDebug:
    score_map_max: ScoreMap
    score_map_min: ScoreMap
    char_overlapped_mask: Mask
    char_neutralized_score_map: ScoreMap
    neutralized_mask: Mask
    neutralized_score_map: ScoreMap


class CharHeatmapDefaultEngine(
    Engine[
        CharHeatmapDefaultEngineInitConfig,
        NoneTypeEngineInitResource,
        CharHeatmapEngineRunConfig,
        CharHeatmap,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'default'

    def generate_np_gaussian_map(self):
        char_radius = self.init_config.gaussian_map_char_radius
        side_length = char_radius * 2

        # Build distances to the center point.
        np_offset = np.abs(np.arange(side_length, dtype=np.float32) - char_radius)
        np_vert_offset = np.repeat(np_offset[:, None], side_length, axis=1)
        np_hori_offset = np.repeat(np_offset[None, :], side_length, axis=0)
        np_distance = np.sqrt(np.square(np_vert_offset) + np.square(np_hori_offset))

        np_norm_distance = np_distance / char_radius
        np_gaussian_map = np.exp(
            -0.5 * np.square(self.init_config.gaussian_map_distance_factor * np_norm_distance)
        )

        # For perspective transformation.
        char_begin = 0
        char_end = side_length - 1
        np_char_src_points = np.asarray(
            [
                (char_begin, char_begin),
                (char_end, char_begin),
                (char_end, char_end),
                (char_begin, char_end),
            ],
            dtype=np.float32,
        )

        return np_gaussian_map, np_char_src_points

    def __init__(
        self,
        init_config: CharHeatmapDefaultEngineInitConfig,
        init_resource: Optional[NoneTypeEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        self.np_gaussian_map, self.np_char_points = self.generate_np_gaussian_map()

    def run(self, run_config: CharHeatmapEngineRunConfig, rng: RandomGenerator) -> CharHeatmap:
        height = run_config.height
        width = run_config.width
        char_polygons = run_config.char_polygons

        shape = (height, width)

        # Intermediate score maps.
        score_map_max = ScoreMap.from_shape(shape)
        score_map_min = ScoreMap.from_shape(shape, value=1.0)

        for char_polygon in char_polygons:
            # Get transformation matrix.
            np_trans_mat = cv.getPerspectiveTransform(
                self.np_char_points,
                char_polygon.internals.np_self_relative_points,
                cv.DECOMP_SVD,
            )

            # Transform gaussian map.
            char_bounding_box = char_polygon.bounding_box

            np_gaussian_map = cv.warpPerspective(
                self.np_gaussian_map,
                np_trans_mat,
                (char_bounding_box.width, char_bounding_box.height),
            )
            score_map = ScoreMap(mat=np_gaussian_map, box=char_bounding_box)

            # Fill score maps.
            char_polygon.fill_score_map(score_map_max, score_map, keep_max_value=True)
            char_polygon.fill_score_map(score_map_min, score_map, keep_min_value=True)

        # Set the char overlapped area while preserving score gte threshold.
        char_overlapped_mask = Mask.from_polygons(
            shape,
            char_polygons,
            ElementSetOperationMode.INTERSECT,
        )

        preserving_score_min = self.init_config.gaussian_map_preserving_score_min
        preserving_mask = Mask(mat=(score_map_max.mat >= preserving_score_min).astype(np.uint8))

        neutralized_mask = Mask.from_masks(
            shape,
            [
                char_overlapped_mask,
                preserving_mask.to_inverted_mask(),
            ],
            ElementSetOperationMode.INTERSECT,
        )

        # Estimate the neutralized score.
        np_delta: np.ndarray = score_map_max.mat - score_map_min.mat  # type: ignore
        np_delta = np.clip(np_delta, 0.0, 1.0)
        char_neutralized_score_map = ScoreMap(mat=np_delta)

        neutralized_score_map = score_map_max.copy()
        neutralized_mask.fill_score_map(neutralized_score_map, char_neutralized_score_map)

        weight = self.init_config.weight_neutralized_score_map
        score_map = ScoreMap(
            mat=((1 - weight) * score_map_max.mat + weight * neutralized_score_map.mat)
        )

        debug = None
        if run_config.enable_debug:
            debug = CharHeatmapDefaultDebug(
                score_map_max=score_map_max,
                score_map_min=score_map_min,
                char_overlapped_mask=char_overlapped_mask,
                char_neutralized_score_map=char_neutralized_score_map,
                neutralized_mask=neutralized_mask,
                neutralized_score_map=neutralized_score_map,
            )

        return CharHeatmap(score_map=score_map, debug=debug)


char_heatmap_default_engine_executor_factory = EngineExecutorFactory(CharHeatmapDefaultEngine)
