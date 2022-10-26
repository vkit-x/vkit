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
from numpy.random import default_rng
from vkit.mechanism.distortion_policy.photometric.color import *
from vkit.mechanism.distortion_policy.photometric.blur import *
from vkit.mechanism.distortion_policy.photometric.noise import *
from vkit.mechanism.distortion_policy.photometric.effect import *
from vkit.mechanism.distortion_policy.photometric.streak import *
from vkit.mechanism.distortion_policy.geometric.affine import *
from vkit.mechanism.distortion_policy.geometric.mls import *
from vkit.mechanism.distortion_policy.geometric.camera import *
from vkit.mechanism.distortion_policy.random_distortion import *
from tests.opt import read_image, write_image


def generate_level_idx(save_png: bool = False):
    for level in range(1, 11):
        for idx in range(3):
            if not save_png:
                ext = 'jpg'
            else:
                ext = 'png'
            yield level, f'l{level}-{idx}.{ext}'


def run_distortion_policy_test(
    distortion_policy_factory: DistortionPolicyFactory,
    index_as_seed: bool = False,
    save_png: bool = False,
):
    distortion_policy = distortion_policy_factory.create()
    image = read_image('Lenna.png').to_rgb_image()

    for idx, (level, name) in enumerate(generate_level_idx(save_png=save_png)):
        if not index_as_seed:
            rng = default_rng(0)
        else:
            rng = default_rng(idx)
        result = distortion_policy.distort(level=level, image=image, rng=rng)
        assert result.image
        write_image(name, result.image, frames_offset=1)


def test_mean_shift_policy_factory():
    run_distortion_policy_test(mean_shift_policy_factory)


def test_color_shift_policy_factory():
    run_distortion_policy_test(color_shift_policy_factory)


def test_brightness_shift_policy_factory():
    run_distortion_policy_test(brightness_shift_policy_factory)


def test_std_shift_policy_factory():
    run_distortion_policy_test(std_shift_policy_factory, index_as_seed=True)


def test_boundary_equalization_policy_factory():
    run_distortion_policy_test(boundary_equalization_policy_factory)


def test_histogram_equalization_policy_factory():
    run_distortion_policy_test(histogram_equalization_policy_factory)


def test_complement_policy_factory():
    run_distortion_policy_test(complement_policy_factory)


def test_posterization_policy_factory():
    run_distortion_policy_test(posterization_policy_factory)


def test_color_balance_policy_factory():
    run_distortion_policy_test(color_balance_policy_factory)


def test_channel_permutation_policy_factory():
    run_distortion_policy_test(channel_permutation_policy_factory, index_as_seed=True)


def test_gaussian_blur_policy_factory():
    run_distortion_policy_test(gaussian_blur_policy_factory)


def test_defocus_blur_policy_factory():
    run_distortion_policy_test(defocus_blur_policy_factory)


def test_motion_blur_policy_factory():
    run_distortion_policy_test(motion_blur_policy_factory, index_as_seed=True)


def test_glass_blur_policy_factory():
    run_distortion_policy_test(glass_blur_policy_factory)


def test_zoom_in_blur_policy_factory():
    run_distortion_policy_test(zoom_in_blur_policy_factory, index_as_seed=True)


def test_gaussion_noise_policy_factory():
    run_distortion_policy_test(gaussion_noise_policy_factory, index_as_seed=True)


def test_poisson_noise_policy_factory():
    run_distortion_policy_test(poisson_noise_policy_factory, index_as_seed=True)


def test_impulse_noise_policy_factory():
    run_distortion_policy_test(impulse_noise_policy_factory, index_as_seed=True)


def test_speckle_noise_policy_factory():
    run_distortion_policy_test(speckle_noise_policy_factory, index_as_seed=True)


def test_jpeg_quality_policy_factory():
    run_distortion_policy_test(
        jpeg_quality_policy_factory,
        index_as_seed=True,
        save_png=True,
    )


def test_pixelation_policy_factory():
    run_distortion_policy_test(pixelation_policy_factory, index_as_seed=True)


def test_fog_policy_factory():
    run_distortion_policy_test(fog_policy_factory, index_as_seed=True)


def test_line_streak_policy_factory():
    run_distortion_policy_test(line_streak_policy_factory, index_as_seed=True)


def test_rectangle_streak_policy_factory():
    run_distortion_policy_test(rectangle_streak_policy_factory, index_as_seed=True)


def test_ellipse_streak_policy_factory():
    run_distortion_policy_test(ellipse_streak_policy_factory, index_as_seed=True)


def test_shear_hori_policy_factory():
    run_distortion_policy_test(shear_hori_policy_factory, index_as_seed=True)


def test_shear_vert_policy_factory():
    run_distortion_policy_test(shear_vert_policy_factory, index_as_seed=True)


def test_rotate_policy_factory():
    run_distortion_policy_test(rotate_policy_factory, index_as_seed=True)


def test_skew_hori_policy_factory():
    run_distortion_policy_test(skew_hori_policy_factory, index_as_seed=True)


def test_skew_vert_policy_factory():
    run_distortion_policy_test(skew_vert_policy_factory, index_as_seed=True)


def test_similarity_mls_policy_factory():
    run_distortion_policy_test(similarity_mls_policy_factory, index_as_seed=True)


def test_camera_plane_only_policy_factory():
    run_distortion_policy_test(camera_plane_only_policy_factory, index_as_seed=True)


def test_camera_cubic_curve_policy_factory():
    run_distortion_policy_test(camera_cubic_curve_policy_factory, index_as_seed=True)


def test_camera_plane_line_fold_policy_factory():
    run_distortion_policy_test(camera_plane_line_fold_policy_factory, index_as_seed=True)


def test_camera_plane_line_curve_policy_factory():
    run_distortion_policy_test(camera_plane_line_curve_policy_factory, index_as_seed=True)


def test_random_distortion():
    random_distortion = random_distortion_factory.create({'force_post_rotate': True})

    for stage in random_distortion.stages:
        print('config:', stage.config)
        print('probs:', stage.distortion_policy_probs)

    image = read_image('Lenna.png').to_rgb_image()
    rng = default_rng(0)

    for idx in range(50):
        debug = RandomDistortionDebug()
        result = random_distortion.distort(image=image, rng=rng, debug=debug)
        assert result.image
        level = min(debug.distortion_levels or (0,))
        distortion_names = debug.distortion_names
        tag = '-'.join(distortion_names) or 'none'
        write_image(f'{idx}-{level}-{len(distortion_names)}-{tag}.jpg', result.image)
