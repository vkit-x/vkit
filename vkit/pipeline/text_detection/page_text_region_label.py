from typing import Tuple, Sequence

import attrs
from numpy.random import Generator as RandomGenerator
import numpy as np
import cv2 as cv

from vkit.utility import rng_choice
from vkit.element import Polygon, ScoreMap, Image
from ..interface import PipelineStep, PipelineStepFactory
from .page_text_region import PageTextRegionStepOutput


@attrs.define
class PageTextRegionLabelStepConfig:
    gaussian_map_length: int = 45
    num_deviate_points_for_each_char: int = 3


@attrs.define
class PageTextRegionLabelStepInput:
    page_text_region_step_output: PageTextRegionStepOutput


@attrs.define
class PageTextRegionLabelStepOutput:
    page_image: Image
    page_score_map: ScoreMap


class PageTextRegionLabelStep(
    PipelineStep[
        PageTextRegionLabelStepConfig,
        PageTextRegionLabelStepInput,
        PageTextRegionLabelStepOutput,
    ]
):  # yapf: disable

    @staticmethod
    def generate_np_gaussian_map(
        length: int,
        rescale_radius_to_value: float = 2.5,
    ):
        # https://colab.research.google.com/drive/1TQ1-BTisMYZHIRVVNpVwDFPviXYMhT7A
        # Build distances to the center point.
        if length % 2 == 0:
            radius = length / 2 - 0.5
        else:
            radius = length // 2

        np_offset = np.abs(np.arange(length, dtype=np.float32) - radius)
        np_vert_offset = np.repeat(np_offset[:, None], length, axis=1)
        np_hori_offset = np.repeat(np_offset[None, :], length, axis=0)
        np_distance = np.sqrt(np.square(np_vert_offset) + np.square(np_hori_offset))

        # Rescale the radius. If default value 2.5 is used, the value in the circle of
        # gaussian map is approximately 0.044, and the value in corner is
        # approximately 0.002.
        np_distance = rescale_radius_to_value * np_distance / radius
        np_gaussian_map = np.exp(-0.5 * np.square(np_distance))

        # For perspective transformation.
        np_points = np.array(
            [
                (0, 0),
                (length - 1, 0),
                (length - 1, length - 1),
                (0, length - 1),
            ],
            dtype=np.float32,
        )

        return np_gaussian_map, np_points

    def generate_page_score_map(
        self,
        shape: Tuple[int, int],
        page_char_polygons: Sequence[Polygon],
    ):
        score_map = ScoreMap.from_shape(shape)

        # Will be transform to fit each char polygon.
        np_src_gaussian_map, np_src_points = self.generate_np_gaussian_map(
            self.config.gaussian_map_length,
        )

        for polygon in page_char_polygons:
            # Transform.
            bounding_box = polygon.fill_np_array_internals.bounding_box

            shifted_polygon = polygon.fill_np_array_internals.get_shifted_polygon()
            np_dst_points = shifted_polygon.to_np_array().astype(np.float32)

            np_dst_gaussian_map = cv.warpPerspective(
                np_src_gaussian_map,
                cv.getPerspectiveTransform(
                    np_src_points,
                    np_dst_points,
                    cv.DECOMP_SVD,
                ),
                (bounding_box.width, bounding_box.height),
            )

            # Fill.
            polygon.fill_score_map(
                score_map,
                np_dst_gaussian_map,
                keep_max_value=True,
            )

        return score_map

    def run(self, input: PageTextRegionLabelStepInput, rng: RandomGenerator):
        page_text_region_step_output = input.page_text_region_step_output
        page_image = page_text_region_step_output.page_image
        page_char_polygons = page_text_region_step_output.page_char_polygons

        page_score_map = self.generate_page_score_map(
            page_image.shape,
            page_char_polygons,
        )

        return PageTextRegionLabelStepOutput(
            page_image=page_image,
            page_score_map=page_score_map,
        )


page_text_region_label_step_factory = PipelineStepFactory(PageTextRegionLabelStep)
