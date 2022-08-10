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
    Polygon,
    TextPolygon,
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
    conflict_control_keyword_groups: Sequence[Sequence[str]] = ()
    force_random_level: bool = False


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
        level: int,
        rng: RandomGenerator,
        debug: Optional[RandomDistortionDebug] = None,
    ):
        if rng.random() > self.config.prob_enable:
            return distortion_result

        distortion_policies = self.sample_distortion_policies(rng)

        if self.config.force_random_level:
            level = rng.integers(LEVEL_MIN, LEVEL_MAX + 1)

        for distortion_policy in distortion_policies:
            distortion_result = distortion_policy.distort(
                level=level,
                shapable_or_shape=distortion_result.shape,
                image=distortion_result.image,
                mask=distortion_result.mask,
                score_map=distortion_result.score_map,
                point=distortion_result.point,
                points=distortion_result.points,
                polygon=distortion_result.polygon,
                polygons=distortion_result.polygons,
                text_polygon=distortion_result.text_polygon,
                text_polygons=distortion_result.text_polygons,
                rng=rng,
                debug=bool(debug),
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

    def distort(
        self,
        rng: RandomGenerator,
        shapable_or_shape: Optional[Union[Shapable, Tuple[int, int]]] = None,
        image: Optional[Image] = None,
        mask: Optional[Mask] = None,
        score_map: Optional[ScoreMap] = None,
        point: Optional[Point] = None,
        points: Optional[PointList] = None,
        polygon: Optional[Polygon] = None,
        polygons: Optional[Iterable[Polygon]] = None,
        text_polygon: Optional[TextPolygon] = None,
        text_polygons: Optional[Iterable[TextPolygon]] = None,
        debug: Optional[RandomDistortionDebug] = None,
    ):
        # Pack.
        distortion_result = Distortion.initialize_distortion_result(
            shapable_or_shape=shapable_or_shape,
            image=image,
            mask=mask,
            score_map=score_map,
        )
        distortion_result.image = image
        distortion_result.mask = mask
        distortion_result.score_map = score_map
        distortion_result.point = point
        distortion_result.points = points
        distortion_result.polygon = polygon
        if polygons:
            distortion_result.polygons = tuple(polygons)
        distortion_result.text_polygon = text_polygon
        if text_polygons:
            distortion_result.text_polygons = tuple(text_polygons)

        # Apply distortions.
        level = rng.integers(self.level_min, self.level_max + 1)

        for stage in self.stages:
            distortion_result = stage.apply_distortions(
                distortion_result=distortion_result,
                level=level,
                rng=rng,
                debug=debug,
            )

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
        2.0,
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
        1.0,
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
        2.0,
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

    @staticmethod
    def unpack_policy_factories_and_default_weights_sum_pairs(
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

    @staticmethod
    def create_policies_and_policy_weights(
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
            )
        )
        if post_rotate_policy:
            # TODO: need to trim inactive area.
            stage_configs.append(
                RandomDistortionStageConfig(
                    distortion_policies=[post_rotate_policy],
                    distortion_policy_weights=[1.0],
                    prob_enable=1.0,
                    num_distortions_min=1,
                    num_distortions_max=1,
                    force_random_level=True,
                )
            )

        return RandomDistortion(
            configs=stage_configs,
            level_min=config.level_min,
            level_max=config.level_max,
        )


random_distortion_factory = RandomDistortionFactory()
