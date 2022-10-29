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
from typing import (
    Mapping,
    Any,
    Iterable,
    Union,
    Tuple,
    Optional,
    Sequence,
    List,
)
from collections import defaultdict
import logging

import attrs
from numpy.random import Generator as RandomGenerator

from vkit.utility import (
    dyn_structure,
    rng_choice_with_size,
    normalize_to_probs,
    PathType,
)
from vkit.element import (
    Shapable,
    Image,
    Point,
    PointList,
    PointTuple,
    Box,
    Polygon,
    Mask,
    ScoreMap,
)
from ..distortion.interface import Distortion, DistortionResult
from .opt import LEVEL_MIN, LEVEL_MAX
from .type import DistortionPolicy, DistortionPolicyFactory
from .photometric import (
    color,
    blur,
    noise,
    effect,
    streak,
)
from .geometric import (
    affine,
    mls,
    camera,
)

logger = logging.getLogger(__name__)


@attrs.define
class RandomDistortionDebug:
    distortion_names: List[str] = attrs.field(factory=list)
    distortion_levels: List[int] = attrs.field(factory=list)
    distortion_images: List[Image] = attrs.field(factory=list)
    distortion_configs: List[Any] = attrs.field(factory=list)
    distortion_states: List[Any] = attrs.field(factory=list)


@attrs.define
class RandomDistortionStageConfig:
    distortion_policies: Sequence[DistortionPolicy]
    distortion_policy_weights: Sequence[float]
    prob_enable: float
    num_distortions_min: int
    num_distortions_max: int
    inject_corner_points: bool = False
    conflict_control_keyword_groups: Sequence[Sequence[str]] = ()
    force_sample_level_in_full_range: bool = False


class RandomDistortionStage:

    def __init__(self, config: RandomDistortionStageConfig):
        self.config = config
        self.distortion_policy_probs = normalize_to_probs(self.config.distortion_policy_weights)

    def sample_distortion_policies(self, rng: RandomGenerator) -> Sequence[DistortionPolicy]:
        num_distortions = rng.integers(
            self.config.num_distortions_min,
            self.config.num_distortions_max + 1,
        )
        if num_distortions <= 0:
            return ()

        num_retries = 5
        while num_retries > 0:
            distortion_policies = rng_choice_with_size(
                rng,
                self.config.distortion_policies,
                size=num_distortions,
                probs=self.distortion_policy_probs,
                replace=False,
            )

            # Conflict analysis.
            conflict_idx_to_count = defaultdict(int)
            for distortion_policy in distortion_policies:
                for conflict_idx, keywords in \
                        enumerate(self.config.conflict_control_keyword_groups):
                    match = False
                    for keyword in keywords:
                        if keyword in distortion_policy.name:
                            match = True
                            break
                    if match:
                        conflict_idx_to_count[conflict_idx] += 1
                        break

            no_conflict = True
            for count in conflict_idx_to_count.values():
                if count > 1:
                    no_conflict = False
                    logger.debug(
                        'distortion policies conflict detected '
                        f'conflict_idx_to_count={conflict_idx_to_count}'
                    )
                    break

            if no_conflict:
                return distortion_policies
            else:
                num_retries -= 1

        logger.warning(f'Cannot sample distortion policies with num_distortion={num_distortions}.')
        return ()

    def apply_distortions(
        self,
        distortion_result: DistortionResult,
        level_min: int,
        level_max: int,
        rng: RandomGenerator,
        debug: Optional[RandomDistortionDebug] = None,
    ):
        if rng.random() > self.config.prob_enable:
            return distortion_result

        if self.config.inject_corner_points:
            height, width = distortion_result.shape

            step = min(height // 4, width // 4)
            assert step > 0

            ys = list(range(0, height, step))
            if ys[-1] < height - 1:
                ys.append(height - 1)

            xs = list(range(0, width, step))
            if xs[0] == 0:
                xs.pop(0)
            if xs[-1] == width - 1:
                xs.pop()

            corner_points = PointList()

            for x in (0, width - 1):
                for y in ys:
                    corner_points.append(Point.create(y=y, x=x))
            for y in (0, height - 1):
                for x in xs:
                    corner_points.append(Point.create(y=y, x=x))

            distortion_result.corner_points = corner_points.to_point_tuple()

        if self.config.force_sample_level_in_full_range:
            level_min = LEVEL_MIN
            level_max = LEVEL_MAX

        distortion_policies = self.sample_distortion_policies(rng)

        for distortion_policy in distortion_policies:
            level = rng.integers(level_min, level_max + 1)

            distortion_result = distortion_policy.distort(
                level=level,
                shapable_or_shape=distortion_result.shape,
                image=distortion_result.image,
                mask=distortion_result.mask,
                score_map=distortion_result.score_map,
                point=distortion_result.point,
                points=distortion_result.points,
                corner_points=distortion_result.corner_points,
                polygon=distortion_result.polygon,
                polygons=distortion_result.polygons,
                rng=rng,
                enable_debug=bool(debug),
            )

            if debug:
                assert distortion_result.image
                debug.distortion_images.append(distortion_result.image)
                debug.distortion_names.append(distortion_policy.name)
                debug.distortion_levels.append(level)
                debug.distortion_configs.append(distortion_result.config)
                debug.distortion_states.append(distortion_result.state)

            distortion_result.config = None
            distortion_result.state = None

        return distortion_result


class RandomDistortion:

    def __init__(
        self,
        configs: Sequence[RandomDistortionStageConfig],
        level_min: int,
        level_max: int,
    ):
        self.stages = [RandomDistortionStage(config) for config in configs]
        self.level_min = level_min
        self.level_max = level_max

    @classmethod
    def get_distortion_result_all_points(cls, distortion_result: DistortionResult):
        if distortion_result.corner_points:
            yield from distortion_result.corner_points

        if distortion_result.point:
            yield distortion_result.point

        if distortion_result.points:
            yield from distortion_result.points

        if distortion_result.polygon:
            yield from distortion_result.polygon.points

        if distortion_result.polygons:
            for polygon in distortion_result.polygons:
                yield from polygon.points

    @classmethod
    def get_distortion_result_element_bounding_box(cls, distortion_result: DistortionResult):
        assert distortion_result.corner_points

        all_points = cls.get_distortion_result_all_points(distortion_result)
        point = next(all_points)
        y_min = point.y
        y_max = point.y
        x_min = point.x
        x_max = point.x
        for point in all_points:
            y_min = min(y_min, point.y)
            y_max = max(y_max, point.y)
            x_min = min(x_min, point.x)
            x_max = max(x_max, point.x)
        return Box(up=y_min, down=y_max, left=x_min, right=x_max)

    @classmethod
    def trim_distortion_result(cls, distortion_result: DistortionResult):
        # Trim page if need.
        if not distortion_result.corner_points:
            return distortion_result

        height, width = distortion_result.shape
        box = cls.get_distortion_result_element_bounding_box(distortion_result)

        pad_up = box.up
        pad_down = height - 1 - box.down
        # NOTE: accept the rounding error.
        assert pad_up >= -1 and pad_down >= -1

        pad_left = box.left
        pad_right = width - 1 - box.right
        assert pad_left >= -1 and pad_right >= -1

        if pad_up <= 0 and pad_down <= 0 and pad_left <= 0 and pad_right <= 0:
            return distortion_result

        # Deal with rounding error.
        up = max(0, box.up)
        down = min(height - 1, box.down)
        left = max(0, box.left)
        right = min(width - 1, box.right)

        pad_up = max(0, pad_up)
        pad_down = max(0, pad_down)
        pad_left = max(0, pad_left)
        pad_right = max(0, pad_right)

        if distortion_result.image:
            distortion_result.image = distortion_result.image.to_cropped_image(
                up=up,
                down=down,
                left=left,
                right=right,
            )

        if distortion_result.mask:
            distortion_result.mask = distortion_result.mask.to_cropped_mask(
                up=up,
                down=down,
                left=left,
                right=right,
            )

        if distortion_result.score_map:
            distortion_result.score_map = distortion_result.score_map.to_cropped_score_map(
                up=up,
                down=down,
                left=left,
                right=right,
            )

        if distortion_result.point:
            distortion_result.point = distortion_result.point.to_shifted_point(
                offset_y=-pad_up,
                offset_x=-pad_left,
            )

        if distortion_result.points:
            distortion_result.points = distortion_result.points.to_shifted_points(
                offset_y=-pad_up,
                offset_x=-pad_left,
            )

        if distortion_result.polygon:
            distortion_result.polygon = distortion_result.polygon.to_shifted_polygon(
                offset_y=-pad_up,
                offset_x=-pad_left,
            )

        if distortion_result.polygons:
            distortion_result.polygons = [
                polygon.to_shifted_polygon(
                    offset_y=-pad_up,
                    offset_x=-pad_left,
                ) for polygon in distortion_result.polygons
            ]

        return distortion_result

    def distort(
        self,
        rng: RandomGenerator,
        shapable_or_shape: Optional[Union[Shapable, Tuple[int, int]]] = None,
        image: Optional[Image] = None,
        mask: Optional[Mask] = None,
        score_map: Optional[ScoreMap] = None,
        point: Optional[Point] = None,
        points: Optional[Union[PointList, PointTuple, Iterable[Point]]] = None,
        polygon: Optional[Polygon] = None,
        polygons: Optional[Iterable[Polygon]] = None,
        debug: Optional[RandomDistortionDebug] = None,
    ):
        # Pack.
        shape = Distortion.get_shape(
            shapable_or_shape=shapable_or_shape,
            image=image,
            mask=mask,
            score_map=score_map,
        )
        distortion_result = DistortionResult(shape=shape)
        distortion_result.image = image
        distortion_result.mask = mask
        distortion_result.score_map = score_map
        distortion_result.point = point
        distortion_result.points = PointTuple(points) if points else None
        distortion_result.polygon = polygon
        if polygons:
            distortion_result.polygons = tuple(polygons)

        # Apply distortions.
        for stage in self.stages:
            distortion_result = stage.apply_distortions(
                distortion_result=distortion_result,
                level_min=self.level_min,
                level_max=self.level_max,
                rng=rng,
                debug=debug,
            )

        distortion_result = self.trim_distortion_result(distortion_result)

        return distortion_result


@attrs.define
class RandomDistortionFactoryConfig:
    # Photometric.
    prob_photometric: float = 1.0
    num_photometric_min: int = 0
    num_photometric_max: int = 2
    photometric_conflict_control_keyword_groups: Sequence[Sequence[str]] = attrs.field(
        factory=lambda: [
            [
                'blur',
                'pixelation',
                'jpeg',
            ],
            [
                'noise',
            ],
        ]
    )
    # Geometric.
    prob_geometric: float = 0.75
    force_post_rotate: bool = False
    # Shared.
    level_min: int = LEVEL_MIN
    level_max: int = LEVEL_MAX
    disabled_policy_names: Sequence[str] = attrs.field(factory=list)
    name_to_policy_config: Mapping[str, Any] = attrs.field(factory=dict)
    name_to_policy_weight: Mapping[str, float] = attrs.field(factory=dict)


_PHOTOMETRIC_POLICY_FACTORIES_AND_DEFAULT_WEIGHTS_SUM_PAIRS = (
    (
        (
            color.mean_shift_policy_factory,
            color.color_shift_policy_factory,
            color.brightness_shift_policy_factory,
            color.std_shift_policy_factory,
            color.boundary_equalization_policy_factory,
            color.histogram_equalization_policy_factory,
            color.complement_policy_factory,
            color.posterization_policy_factory,
            color.color_balance_policy_factory,
            color.channel_permutation_policy_factory,
        ),
        10.0,
    ),
    (
        (
            blur.gaussian_blur_policy_factory,
            blur.defocus_blur_policy_factory,
            blur.motion_blur_policy_factory,
            blur.glass_blur_policy_factory,
            blur.zoom_in_blur_policy_factory,
        ),
        1.0,
    ),
    (
        (
            noise.gaussion_noise_policy_factory,
            noise.poisson_noise_policy_factory,
            noise.impulse_noise_policy_factory,
            noise.speckle_noise_policy_factory,
        ),
        3.0,
    ),
    (
        (
            effect.jpeg_quality_policy_factory,
            effect.pixelation_policy_factory,
            effect.fog_policy_factory,
        ),
        1.0,
    ),
    (
        (
            streak.line_streak_policy_factory,
            streak.rectangle_streak_policy_factory,
            streak.ellipse_streak_policy_factory,
        ),
        1.0,
    ),
)

_GEOMETRIC_POLICY_FACTORIES_AND_DEFAULT_WEIGHTS_SUM_PAIRS = (
    (
        (
            affine.shear_hori_policy_factory,
            affine.shear_vert_policy_factory,
            affine.rotate_policy_factory,
            affine.skew_hori_policy_factory,
            affine.skew_vert_policy_factory,
        ),
        1.0,
    ),
    (
        (mls.similarity_mls_policy_factory,),
        1.0,
    ),
    (
        (
            camera.camera_plane_only_policy_factory,
            camera.camera_cubic_curve_policy_factory,
            camera.camera_plane_line_fold_policy_factory,
            camera.camera_plane_line_curve_policy_factory,
        ),
        1.0,
    ),
)


class RandomDistortionFactory:

    @classmethod
    def unpack_policy_factories_and_default_weights_sum_pairs(
        cls,
        policy_factories_and_default_weights_sum_pairs: Sequence[
            Tuple[
                Sequence[DistortionPolicyFactory],
                float,
            ]
        ]
    ):  # yapf: disable
        flatten_policy_factories: List[DistortionPolicyFactory] = []
        flatten_policy_default_weights: List[float] = []

        for policy_factories, default_weights_sum in policy_factories_and_default_weights_sum_pairs:
            default_weight = default_weights_sum / len(policy_factories)
            flatten_policy_factories.extend(policy_factories)
            flatten_policy_default_weights.extend([default_weight] * len(policy_factories))

        assert len(flatten_policy_factories) == len(flatten_policy_default_weights)
        return flatten_policy_factories, flatten_policy_default_weights

    def __init__(
        self,
        photometric_policy_factories_and_default_weights_sum_pairs: Sequence[
            Tuple[
                Sequence[DistortionPolicyFactory],
                float,
            ]
        ] = _PHOTOMETRIC_POLICY_FACTORIES_AND_DEFAULT_WEIGHTS_SUM_PAIRS,
        geometric_policy_factories_and_default_weights_sum_pairs: Sequence[
            Tuple[
                Sequence[DistortionPolicyFactory],
                float,
            ]
        ] = _GEOMETRIC_POLICY_FACTORIES_AND_DEFAULT_WEIGHTS_SUM_PAIRS,
    ):  # yapf: disable
        (
            self.photometric_policy_factories,
            self.photometric_policy_default_weights,
        ) = self.unpack_policy_factories_and_default_weights_sum_pairs(
            photometric_policy_factories_and_default_weights_sum_pairs
        )

        (
            self.geometric_policy_factories,
            self.geometric_policy_default_weights,
        ) = self.unpack_policy_factories_and_default_weights_sum_pairs(
            geometric_policy_factories_and_default_weights_sum_pairs
        )

    @classmethod
    def create_policies_and_policy_weights(
        cls,
        policy_factories: Sequence[DistortionPolicyFactory],
        policy_default_weights: Sequence[float],
        config: RandomDistortionFactoryConfig,
    ):
        disabled_policy_names = set(config.disabled_policy_names)

        policies: List[DistortionPolicy] = []
        policy_weights: List[float] = []

        for policy_factory, policy_default_weight in zip(policy_factories, policy_default_weights):
            if policy_factory.name in disabled_policy_names:
                continue

            policy_config = config.name_to_policy_config.get(policy_factory.name)
            policies.append(policy_factory.create(policy_config))

            policy_weight = policy_default_weight
            if policy_factory.name in config.name_to_policy_weight:
                policy_weight = config.name_to_policy_weight[policy_factory.name]
            policy_weights.append(policy_weight)

        return policies, policy_weights

    def create(
        self,
        config: Optional[
            Union[
                Mapping[str, Any],
                PathType,
                RandomDistortionFactoryConfig,
            ]
        ] = None,
    ):  # yapf: disable
        config = dyn_structure(
            config,
            RandomDistortionFactoryConfig,
            support_path_type=True,
            support_none_type=True,
        )

        stage_configs: List[RandomDistortionStageConfig] = []

        # Photometric.
        (
            photometric_policies,
            photometric_policy_weights,
        ) = self.create_policies_and_policy_weights(
            self.photometric_policy_factories,
            self.photometric_policy_default_weights,
            config,
        )
        stage_configs.append(
            RandomDistortionStageConfig(
                distortion_policies=photometric_policies,
                distortion_policy_weights=photometric_policy_weights,
                prob_enable=config.prob_photometric,
                num_distortions_min=config.num_photometric_min,
                num_distortions_max=config.num_photometric_max,
                conflict_control_keyword_groups=config.photometric_conflict_control_keyword_groups,
            )
        )

        # Geometric.
        (
            geometric_policies,
            geometric_policy_weights,
        ) = self.create_policies_and_policy_weights(
            self.geometric_policy_factories,
            self.geometric_policy_default_weights,
            config,
        )

        post_rotate_policy = None
        if config.force_post_rotate:
            rotate_policy_idx = -1
            for geometric_policy_idx, geometric_policy in enumerate(geometric_policies):
                if geometric_policy.name == 'rotate':
                    rotate_policy_idx = geometric_policy_idx
                    break
            assert rotate_policy_idx >= 0
            post_rotate_policy = geometric_policies.pop(rotate_policy_idx)
            geometric_policy_weights.pop(rotate_policy_idx)

        stage_configs.append(
            RandomDistortionStageConfig(
                distortion_policies=geometric_policies,
                distortion_policy_weights=geometric_policy_weights,
                prob_enable=config.prob_geometric,
                num_distortions_min=1,
                num_distortions_max=1,
                inject_corner_points=config.force_post_rotate,
            )
        )
        if post_rotate_policy:
            stage_configs.append(
                RandomDistortionStageConfig(
                    distortion_policies=[post_rotate_policy],
                    distortion_policy_weights=[1.0],
                    prob_enable=1.0,
                    num_distortions_min=1,
                    num_distortions_max=1,
                    force_sample_level_in_full_range=True,
                )
            )

        return RandomDistortion(
            configs=stage_configs,
            level_min=config.level_min,
            level_max=config.level_max,
        )


random_distortion_factory = RandomDistortionFactory()
