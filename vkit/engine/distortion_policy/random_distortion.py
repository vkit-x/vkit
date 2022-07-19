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
    rng_choice,
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
from ..distortion.interface import Distortion
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


class RandomDistortion:

    def __init__(
        self,
        photometric_policies: Sequence[DistortionPolicy],
        photometric_policy_weights: Sequence[float],
        num_photometric_min: int,
        num_photometric_max: int,
        geometric_policies: Sequence[DistortionPolicy],
        geometric_policy_weights: Sequence[float],
        prob_geometric: float,
        policy_conflict_control_keyword_groups: Sequence[Sequence[str]],
        level_min: int,
        level_max: int,
    ):
        self.photometric_policies = photometric_policies
        self.photometric_policy_probs = normalize_to_probs(photometric_policy_weights)
        self.num_photometric_min = num_photometric_min
        self.num_photometric_max = num_photometric_max

        self.geometric_policies = geometric_policies
        self.geometric_policy_probs = normalize_to_probs(geometric_policy_weights)
        self.prob_geometric = prob_geometric

        self.policy_conflict_control_keyword_groups = policy_conflict_control_keyword_groups

        self.level_min = level_min
        self.level_max = level_max

    def sample_photometric_policies(self, rng: RandomGenerator) -> Sequence[DistortionPolicy]:
        num_photometric = rng.integers(self.num_photometric_min, self.num_photometric_max + 1)
        if num_photometric <= 0:
            return []

        num_retries = 5
        while num_retries > 0:
            photometric_policies = rng_choice_with_size(
                rng,
                self.photometric_policies,
                size=num_photometric,
                probs=self.photometric_policy_probs,
            )

            # Conflict analysis.
            conflict_idx_to_count = defaultdict(int)
            for photometric_policy in photometric_policies:
                for conflict_idx, keywords in \
                        enumerate(self.policy_conflict_control_keyword_groups):
                    match = False
                    for keyword in keywords:
                        if keyword in photometric_policy.name:
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
                        'photometric policies conflict detected '
                        f'conflict_idx_to_count={conflict_idx_to_count}'
                    )
                    break

            if no_conflict:
                return photometric_policies
            else:
                num_retries -= 1

        logger.warning(
            f'Cannot sample photometric policies with num_photometric={num_photometric}.'
        )
        return []

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
        debug: bool = False,
    ):
        result = Distortion.initialize_distortion_result(
            shapable_or_shape=shapable_or_shape,
            image=image,
            mask=mask,
            score_map=score_map,
        )
        result.image = image
        result.mask = mask
        result.score_map = score_map
        result.point = point
        result.points = points
        result.polygon = polygon
        if polygons:
            result.polygons = tuple(polygons)
        result.text_polygon = text_polygon
        if text_polygons:
            result.text_polygons = tuple(text_polygons)

        distortion_names: List[str] = []
        distortion_images: List[Image] = []
        distortion_configs: List[Any] = []
        distortion_states: List[Any] = []

        level = rng.integers(self.level_min, self.level_max + 1)

        for photometric_policy in self.sample_photometric_policies(rng):
            result = photometric_policy.distort(
                level=level,
                shapable_or_shape=result.shape,
                image=result.image,
                mask=result.mask,
                score_map=result.score_map,
                point=result.point,
                points=result.points,
                polygon=result.polygon,
                polygons=result.polygons,
                text_polygon=result.text_polygon,
                text_polygons=result.text_polygons,
                rng=rng,
                debug=debug,
            )

            distortion_names.append(photometric_policy.name)
            if debug:
                assert result.image
                distortion_images.append(result.image)
                distortion_configs.append(result.config)
                distortion_states.append(result.state)
            result.config = None
            result.state = None

        if rng.random() < self.prob_geometric:
            geometric_policy = rng_choice(
                rng,
                self.geometric_policies,
                probs=self.geometric_policy_probs,
            )
            result = geometric_policy.distort(
                level=level,
                shapable_or_shape=result.shape,
                image=result.image,
                mask=result.mask,
                score_map=result.score_map,
                point=result.point,
                points=result.points,
                polygon=result.polygon,
                polygons=result.polygons,
                text_polygon=result.text_polygon,
                text_polygons=result.text_polygons,
                rng=rng,
                debug=debug
            )

            distortion_names.append(geometric_policy.name)
            if debug:
                assert result.image
                distortion_images.append(result.image)
                distortion_configs.append(result.config)
                distortion_states.append(result.state)
            result.config = None
            result.state = None

        result.meta = {
            'image': image,
            'level': level,
            'distortion_names': distortion_names,
        }
        if debug:
            # Could increase RAM consumption.
            result.meta.update({
                'distortion_images': distortion_images,
                'distortion_configs': distortion_configs,
                'distortion_states': distortion_states,
            })
        return result


@attrs.define
class RandomDistortionFactoryConfig:
    num_photometric_min: int = 0
    num_photometric_max: int = 2
    prob_geometric: float = 0.75
    level_min: int = 1
    level_max: int = 10
    policy_conflict_control_keyword_groups: Sequence[Sequence[str]] = attrs.field(
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
    ((mls.similarity_mls_policy_factory,), 1.0),
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

    def create(
        self,
        config: Optional[Union[Mapping[str, Any], PathType, RandomDistortionFactoryConfig]] = None,
    ):
        config = dyn_structure(
            config,
            RandomDistortionFactoryConfig,
            support_path_type=True,
            support_none_type=True,
        )

        (
            photometric_policies,
            photometric_policy_weights,
        ) = self.create_policies_and_policy_weights(
            self.photometric_policy_factories,
            self.photometric_policy_default_weights,
            config,
        )

        (
            geometric_policies,
            geometric_policy_weights,
        ) = self.create_policies_and_policy_weights(
            self.geometric_policy_factories,
            self.geometric_policy_default_weights,
            config,
        )

        return RandomDistortion(
            photometric_policies=photometric_policies,
            photometric_policy_weights=photometric_policy_weights,
            num_photometric_min=config.num_photometric_min,
            num_photometric_max=config.num_photometric_max,
            geometric_policies=geometric_policies,
            geometric_policy_weights=geometric_policy_weights,
            prob_geometric=config.prob_geometric,
            policy_conflict_control_keyword_groups=config.policy_conflict_control_keyword_groups,
            level_min=config.level_min,
            level_max=config.level_max,
        )


random_distortion_factory = RandomDistortionFactory()
