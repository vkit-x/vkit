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
from typing import Optional, List
import math
import itertools

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv

from vkit.element import Box, Mask
from vkit.mechanism.distortion.geometric.affine import affine_np_points
from ..char_heatmap.default import build_np_distance
from ..interface import (
    Engine,
    EngineExecutorFactory,
    NoneTypeEngineInitResource,
)
from .type import CharMaskEngineRunConfig, CharMask


@attrs.define
class CharMaskExternalEllipseEngineInitConfig:
    internal_side_length: int = 40


class CharMaskExternalEllipseEngine(
    Engine[
        CharMaskExternalEllipseEngineInitConfig,
        NoneTypeEngineInitResource,
        CharMaskEngineRunConfig,
        CharMask,
    ]
):  # yapf: disable

    @classmethod
    def get_type_name(cls) -> str:
        return 'external_ellipse'

    def __init__(
        self,
        init_config: CharMaskExternalEllipseEngineInitConfig,
        init_resource: Optional[NoneTypeEngineInitResource] = None,
    ):
        super().__init__(init_config, init_resource)

        internal_side_length = init_config.internal_side_length
        external_radius = math.ceil(internal_side_length / math.sqrt(2))

        # Build distances to the center point.
        np_distance = build_np_distance(external_radius)

        # Build mask.
        self.np_external_mask = (np_distance <= external_radius).astype(np.uint8)
        external_side_length = self.np_external_mask.shape[0]

        # For perspective transformation.
        char_pad = (external_side_length - internal_side_length) // 2
        char_begin = char_pad
        char_end = char_pad + internal_side_length - 1
        self.np_char_points = np.asarray(
            [
                (char_begin, char_begin),
                (char_end, char_begin),
                (char_end, char_end),
                (char_begin, char_end),
            ],
            dtype=np.float32,
        )

        external_begin = 0
        external_end = external_side_length - 1
        self.np_external_points = np.asarray(
            [
                (external_begin, external_begin),
                (external_end, external_begin),
                (external_end, external_end),
                (external_begin, external_end),
            ],
            dtype=np.float32,
        )

    def run(self, run_config: CharMaskEngineRunConfig, rng: RandomGenerator) -> CharMask:
        char_polygons = run_config.char_polygons
        char_bounding_boxes = run_config.char_bounding_boxes

        if char_bounding_boxes:
            assert len(char_bounding_boxes) == len(char_polygons)
        else:
            bounding_box = Box(
                up=0,
                down=run_config.height - 1,
                left=0,
                right=run_config.width - 1,
            )
            char_bounding_boxes = itertools.repeat(bounding_box)

        combined_chars_mask = Mask.from_shape((run_config.height, run_config.width))
        char_masks: List[Mask] = []

        for char_polygon, char_bounding_box in zip(char_polygons, char_bounding_boxes):
            # 1. Find the transformed external points.
            assert char_polygon.num_points == 4
            np_trans_mat = cv.getPerspectiveTransform(
                self.np_char_points,
                char_polygon.internals.np_self_relative_points,
                cv.DECOMP_SVD,
            )
            np_transformed_external_points = affine_np_points(
                np_trans_mat,
                self.np_external_points,
            )

            # Make self-relative and keep the offset.
            y_offset = np_transformed_external_points[:, 1].min()
            x_offset = np_transformed_external_points[:, 0].min()
            np_transformed_external_points[:, 1] -= y_offset
            np_transformed_external_points[:, 0] -= x_offset

            # 2. Transform the external mask.
            np_transformed_external_points = np_transformed_external_points.astype(np.float32)
            np_trans_mat = cv.getPerspectiveTransform(
                self.np_external_points,
                np_transformed_external_points,
                cv.DECOMP_SVD,
            )
            y_max = np_transformed_external_points[:, 1].max()
            transformed_height = math.ceil(y_max)
            x_max = np_transformed_external_points[:, 0].max()
            transformed_width = math.ceil(x_max)
            np_transformed_external_mask = cv.warpPerspective(
                self.np_external_mask,
                np_trans_mat,
                (transformed_width, transformed_height),
            )

            # 3. Place char mask.
            smooth_y_min = min(point.smooth_y for point in char_polygon.points)
            smooth_x_min = min(point.smooth_x for point in char_polygon.points)

            target_up = round(smooth_y_min + y_offset)
            target_down = target_up + transformed_height - 1
            target_left = round(smooth_x_min + x_offset)
            target_right = target_left + transformed_width - 1

            trimmed_up = 0
            if target_up < char_bounding_box.up:
                trimmed_up = char_bounding_box.up - target_up
                target_up = char_bounding_box.up

            trimmed_down = transformed_height - 1
            if target_down > char_bounding_box.down:
                trimmed_down -= (target_down - char_bounding_box.down)
                target_down = char_bounding_box.down

            trimmed_left = 0
            if target_left < char_bounding_box.left:
                trimmed_left = char_bounding_box.left - target_left
                target_left = char_bounding_box.left

            trimmed_right = transformed_width - 1
            if target_right > char_bounding_box.right:
                trimmed_right -= (target_right - char_bounding_box.right)
                target_right = char_bounding_box.right

            target_box = Box(
                up=target_up,
                down=target_down,
                left=target_left,
                right=target_right,
            )
            np_transformed_external_mask = \
                np_transformed_external_mask[
                    trimmed_up:trimmed_down + 1,
                    trimmed_left:trimmed_right + 1
                ]
            char_mask = Mask(mat=np_transformed_external_mask, box=target_box)
            char_masks.append(char_mask)

            # Fill.
            char_mask.fill_mask(combined_chars_mask, 1, keep_max_value=True)

        return CharMask(
            combined_chars_mask=combined_chars_mask,
            char_masks=char_masks,
        )


char_mask_external_ellipse_engine_executor_factory = EngineExecutorFactory(
    CharMaskExternalEllipseEngine
)
