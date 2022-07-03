from typing import Sequence, Tuple, Optional

import numpy as np
from numpy.random import Generator as RandomGenerator
import attrs

from vkit.element import Point
from ..interface import DistortionConfig
from .grid_rendering.interface import (
    PointProjector,
    DistortionStateImageGridBased,
    DistortionImageGridBased,
)
from .grid_rendering.grid_creator import create_src_image_grid


@attrs.define
class SimilarityMlsConfig(DistortionConfig):
    src_handle_points: Sequence[Point]
    dst_handle_points: Sequence[Point]
    grid_size: int
    resize_as_src: bool = False


class SimilarityMlsPointProjector(PointProjector):

    def __init__(self, src_handle_points: Sequence[Point], dst_handle_points: Sequence[Point]):
        self.src_handle_points = src_handle_points
        self.dst_handle_points = dst_handle_points

        self.src_xy_pair_to_dst_point = {
            (src_point.x, src_point.y): dst_point
            for src_point, dst_point in zip(src_handle_points, dst_handle_points)
        }

        self.src_handle_np_points = np.asarray(
            [(point.x, point.y) for point in src_handle_points],
            dtype=np.int32,
        )
        self.dst_handle_np_points = np.asarray(
            [(point.x, point.y) for point in dst_handle_points],
            dtype=np.int32,
        )

    def project_point(self, src_point: Point):
        '''
        Calculate the corresponding dst point given the src point.
        Paper: https://people.engr.tamu.edu/schaefer/research/mls.pdf
        '''
        src_xy_pair = (src_point.x, src_point.y)

        if src_xy_pair in self.src_xy_pair_to_dst_point:
            # Identity.
            # NOTE: copy is important since this point could be changed later.
            # TODO: re-think the immutable design.
            return self.src_xy_pair_to_dst_point[src_xy_pair].copy()

        # Calculate the distance to src handles.
        src_distance_squares = self.src_handle_np_points.copy()
        src_distance_squares[:, 0] -= src_point.x
        src_distance_squares[:, 1] -= src_point.y
        np.square(src_distance_squares, out=src_distance_squares)
        # (N), and should not contain 0.0.
        src_distance_squares = np.sum(src_distance_squares, axis=1)

        # Calculate weights based on distances.
        # (N), and should not contain inf.
        with np.errstate(divide='raise'):
            src_distance_squares_inverse = 1 / src_distance_squares
            weights = src_distance_squares_inverse / np.sum(src_distance_squares_inverse)

        # (2), the weighted centroids.
        src_centroid = np.matmul(weights, self.src_handle_np_points)
        dst_centroid = np.matmul(weights, self.dst_handle_np_points)

        # (N, 2)
        src_hat = self.src_handle_np_points - src_centroid
        dst_hat = self.dst_handle_np_points - dst_centroid

        # (N, 2)
        src_hat_vert = src_hat[:, [1, 0]]
        src_hat_vert[:, 0] *= -1

        # Calculate matrix A.
        src_centroid_x, src_centroid_y = src_centroid
        src_mat_anchor = np.transpose(
            np.array(
                [
                    # v - p*
                    (
                        src_point.x - src_centroid_x,
                        src_point.y - src_centroid_y,
                    ),
                    # -(v - p*)^vert
                    (
                        src_point.y - src_centroid_y,
                        -(src_point.x - src_centroid_x),
                    ),
                ],
                dtype=np.float32,
            )
        )
        # (N, 2)
        src_mat_row0 = np.matmul(src_hat, src_mat_anchor)
        src_mat_row1 = np.matmul(-src_hat_vert, src_mat_anchor)
        # (N, 2, 2)
        src_mat = (
            np.expand_dims(np.expand_dims(src_distance_squares_inverse, axis=1), axis=1)
            * np.stack((src_mat_row0, src_mat_row1), axis=1)
        )

        # Calculate the point in dst.
        # (N, 2)
        dst_prod = np.squeeze(
            # (N, 1, 2)
            np.matmul(
                # (N, 1, 2)
                np.expand_dims(dst_hat, axis=1),
                # (N, 2, 2)
                src_mat,
            ),
            axis=1,
        )
        mu = np.sum(src_distance_squares_inverse * np.sum(src_hat * src_hat, axis=1))
        dst_x, dst_y = np.sum(dst_prod, axis=0) / mu + dst_centroid

        return Point(y=round(dst_y), x=round(dst_x))


class SimilarityMlsState(DistortionStateImageGridBased[SimilarityMlsConfig]):

    def __init__(
        self,
        config: SimilarityMlsConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        height, width = shape
        self.initialize_image_grid_based(
            create_src_image_grid(height, width, config.grid_size),
            SimilarityMlsPointProjector(
                config.src_handle_points,
                config.dst_handle_points,
            ),
            resize_as_src=config.resize_as_src,
        )

        # For debug only.
        self.dst_handle_points = list(map(self.shift_and_resize_point, config.dst_handle_points))


similarity_mls = DistortionImageGridBased(
    config_cls=SimilarityMlsConfig,
    state_cls=SimilarityMlsState,
)
