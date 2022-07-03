from typing import Tuple, List

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import Point
from vkit.engine import distortion
from ..type import DistortionConfigGenerator, DistortionPolicyFactory
from ..opt import sample_float, SampleFloatMode, generate_grid_size


@attrs.define
class SimilarityMlsConfigGeneratorConfig:
    num_segments_min: int = 2
    num_segments_max: int = 4
    step_min: int = 10
    radius_max_ratio_min: float = 0.025
    radius_max_ratio_max: float = 0.125
    grid_size_min: int = 15
    grid_size_ratio: float = 0.01


class SimilarityMlsConfigGenerator(
    DistortionConfigGenerator[
        SimilarityMlsConfigGeneratorConfig,
        distortion.SimilarityMlsConfig,
    ]
):  # yapf: disable

    @staticmethod
    def generate_coord(length: int, step: int, rng: RandomGenerator):
        end = length - 1
        if end % step == 0:
            steps = [step] * (end // step)
        else:
            steps = [step] * (end // step - 1)
            steps.append(step + end % step)
        assert sum(steps) == end

        rng.shuffle(steps)
        coord: List[int] = [0]
        for step in steps:
            pos = coord[-1] + step
            coord.append(pos)
        return coord

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator):
        # Generate control points.
        short_side_length = min(shape)
        num_segments = rng.integers(self.config.num_segments_min, self.config.num_segments_max + 1)
        step = (short_side_length - 1) // num_segments
        if step < self.config.step_min:
            # Downgrade to corners if the gap is too small.
            step = short_side_length - 1

        height, width = shape
        # NOTE:
        # 1. Corners are always included.
        # 2. Distance of any two points >= step.
        coord_y = SimilarityMlsConfigGenerator.generate_coord(height, step, rng)
        coord_x = SimilarityMlsConfigGenerator.generate_coord(width, step, rng)
        src_handle_points: List[Point] = []
        for y in coord_y:
            for x in coord_x:
                src_handle_points.append(Point(y=y, x=x))

        # Generate deformed points.
        assert self.config.radius_max_ratio_max < 0.5
        radius_max_ratio = sample_float(
            level=self.level,
            value_min=self.config.radius_max_ratio_min,
            value_max=self.config.radius_max_ratio_max,
            prob_reciprocal=None,
            rng=rng,
            mode=SampleFloatMode.QUAD,
        )
        radius = int(radius_max_ratio * step)
        dst_handle_points: List[Point] = []
        for point in src_handle_points:
            delta_y = rng.integers(-radius, radius + 1)
            delta_x = rng.integers(-radius, radius + 1)
            dst_handle_points.append(Point(
                y=point.y + delta_y,
                x=point.x + delta_x,
            ))

        # Generate grid size.
        grid_size = generate_grid_size(
            self.config.grid_size_min,
            self.config.grid_size_ratio,
            shape,
        )

        return distortion.SimilarityMlsConfig(
            src_handle_points=src_handle_points,
            dst_handle_points=dst_handle_points,
            grid_size=grid_size,
        )


similarity_mls_policy_factory = DistortionPolicyFactory(
    distortion.similarity_mls,
    SimilarityMlsConfigGenerator,
)
