from typing import (
    Dict,
    Any,
    Iterable,
    Union,
    Tuple,
    Optional,
    Sequence,
    List,
    Set,
)

import attrs
from numpy.random import RandomState

from vkit.utility import (
    dyn_structure,
    rnd_choice,
    rnd_choice_with_size,
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


class RandomDistortion:

    def __init__(
        self,
        photometric_policies: Sequence[DistortionPolicy],
        num_photometric_min: int,
        num_photometric_max: int,
        geometric_policies: Sequence[DistortionPolicy],
        prob_geometric: float,
        policy_conflict_control_keyword_groups: Sequence[Sequence[str]],
        level_min: int,
        level_max: int,
    ):
        self.photometric_policies = photometric_policies
        self.num_photometric_min = num_photometric_min
        self.num_photometric_max = num_photometric_max

        self.geometric_policies = geometric_policies
        self.prob_geometric = prob_geometric

        self.policy_conflict_control_keyword_groups = policy_conflict_control_keyword_groups

        self.level_min = level_min
        self.level_max = level_max

    def sample_photometric_policies(self, rnd: RandomState) -> Sequence[DistortionPolicy]:
        num_photometric = rnd.randint(self.num_photometric_min, self.num_photometric_max + 1)
        if num_photometric <= 0:
            return []

        photometric_policies: List[DistortionPolicy] = []
        conflicted_indices: Set[int] = set()

        for photometric_policy in rnd_choice_with_size(
            rnd,
            self.photometric_policies,
            size=num_photometric,
        ):
            skip = False
            for idx, keywords in enumerate(self.policy_conflict_control_keyword_groups):
                match = False
                for keyword in keywords:
                    if keyword in photometric_policy.name:
                        match = True
                        break
                if match:
                    if idx in conflicted_indices:
                        skip = True
                    else:
                        conflicted_indices.add(idx)
                    break

            if skip:
                continue

            photometric_policies.append(photometric_policy)

        return photometric_policies

    def distort(
        self,
        rnd: RandomState,
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

        level = rnd.randint(self.level_min, self.level_max + 1)

        for photometric_policy in self.sample_photometric_policies(rnd):
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
                rnd=rnd,
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

        if rnd.random() < self.prob_geometric:
            geometric_policy = rnd_choice(rnd, self.geometric_policies)
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
                rnd=rnd,
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
    name_to_policy_config: Dict[str, Any] = attrs.field(factory=dict)


class RandomDistortionFactory:

    def __init__(
        self,
        photometric_policy_factories: Sequence[DistortionPolicyFactory] = (
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
            blur.gaussian_blur_policy_factory,
            blur.defocus_blur_policy_factory,
            blur.motion_blur_policy_factory,
            blur.glass_blur_policy_factory,
            blur.zoom_in_blur_policy_factory,
            noise.gaussion_noise_policy_factory,
            noise.poisson_noise_policy_factory,
            noise.impulse_noise_policy_factory,
            noise.speckle_noise_policy_factory,
            effect.jpeg_quality_policy_factory,
            effect.pixelation_policy_factory,
            effect.fog_policy_factory,
            streak.line_streak_policy_factory,
            streak.rectangle_streak_policy_factory,
            streak.ellipse_streak_policy_factory,
        ),
        geometric_policy_factories: Sequence[DistortionPolicyFactory] = (
            affine.shear_hori_policy_factory,
            affine.shear_vert_policy_factory,
            affine.rotate_policy_factory,
            affine.skew_hori_policy_factory,
            affine.skew_vert_policy_factory,
            mls.similarity_mls_policy_factory,
            camera.camera_plane_only_policy_factory,
            camera.camera_cubic_curve_policy_factory,
            camera.camera_plane_line_fold_policy_factory,
            camera.camera_plane_line_curve_policy_factory,
        ),
    ):
        self.photometric_policy_factories = photometric_policy_factories
        self.geometric_policy_factories = geometric_policy_factories

    def create(
        self,
        config: Optional[Union[Dict[str, Any], PathType, RandomDistortionFactoryConfig]] = None,
    ):
        config = dyn_structure(
            config,
            RandomDistortionFactoryConfig,
            support_path_type=True,
            support_none_type=True,
        )

        disabled_policy_names = set(config.disabled_policy_names)

        photometric_policies: List[DistortionPolicy] = []
        for policy_factory in self.photometric_policy_factories:
            if policy_factory.name in disabled_policy_names:
                continue
            policy_config = config.name_to_policy_config.get(policy_factory.name)
            photometric_policies.append(policy_factory.create(policy_config))

        geometric_policies: List[DistortionPolicy] = []
        for policy_factory in self.geometric_policy_factories:
            if policy_factory.name in disabled_policy_names:
                continue
            policy_config = config.name_to_policy_config.get(policy_factory.name)
            geometric_policies.append(policy_factory.create(policy_config))

        return RandomDistortion(
            photometric_policies=photometric_policies,
            num_photometric_min=config.num_photometric_min,
            num_photometric_max=config.num_photometric_max,
            geometric_policies=geometric_policies,
            prob_geometric=config.prob_geometric,
            policy_conflict_control_keyword_groups=config.policy_conflict_control_keyword_groups,
            level_min=config.level_min,
            level_max=config.level_max,
        )


random_distortion_factory = RandomDistortionFactory()
