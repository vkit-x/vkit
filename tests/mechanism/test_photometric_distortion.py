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
import numpy as np
from numpy.random import default_rng

from vkit.mechanism.distortion import *
from tests.opt import read_image, write_image


def test_mean_shift():
    image = read_image('Lenna.png').to_rgb_image()
    mean = np.mean(image.mat.reshape(-1, 3), axis=0)

    dst_image = mean_shift.distort_image(
        {'delta': 100},
        image,
    )
    dst_mean = np.mean(dst_image.mat.reshape(-1, 3), axis=0)
    assert (np.abs(dst_mean - mean) > 50).all()
    write_image('default.jpg', dst_image)

    for channel in [0, 1, 2]:
        dst_image = mean_shift.distort_image(
            MeanShiftConfig(delta=100, channels=[channel]),
            image,
        )
        dst_mean = np.mean(dst_image.mat.reshape(-1, 3), axis=0)
        mean_delta = np.abs(dst_mean - mean)
        assert dst_mean[channel] != mean[channel]
        assert (mean_delta == 0).sum() == 2
        write_image(f'channel_{channel}.jpg', dst_image)

    dst_image = mean_shift.distort_image(
        MeanShiftConfig(delta=255),
        image,
    )
    dst_mean = np.mean(dst_image.mat.reshape(-1, 3), axis=0)
    assert (dst_mean == 255).all()
    write_image('clip.jpg', dst_image)

    dst_image = mean_shift.distort_image(
        MeanShiftConfig(delta=256, oob_behavior=OutOfBoundBehavior.CYCLE),
        image,
    )
    dst_mean = np.mean(dst_image.mat.reshape(-1, 3), axis=0)
    assert (dst_mean == mean).all()
    write_image('cycle.jpg', dst_image)

    dst_image = mean_shift.distort_image(
        MeanShiftConfig(delta=128, oob_behavior=OutOfBoundBehavior.CYCLE),
        image,
    )
    dst_mean = np.mean(dst_image.mat.reshape(-1, 3), axis=0)
    assert (dst_mean != mean).all()
    write_image('cycle_half.jpg', dst_image)

    dst_image = mean_shift.distort_image(
        MeanShiftConfig(delta=128, oob_behavior=OutOfBoundBehavior.CYCLE, threshold=127),
        image,
    )
    assert dst_image.mat.min() >= 128
    write_image('threshold.jpg', dst_image)

    dst_image = mean_shift.distort_image(
        MeanShiftConfig(delta=-128, oob_behavior=OutOfBoundBehavior.CYCLE, threshold=128),
        image,
    )
    assert dst_image.mat.min() <= 127
    write_image('threshold_neg.jpg', dst_image)


def test_color_shift():
    image = read_image('Lenna.png').to_hsl_image()

    for delta in [50, 100, 150, 200, 250]:
        dst_image = color_shift.distort_image(ColorShiftConfig(delta), image)
        assert (dst_image.mat[:, :, [1, 2]] == image.mat[:, :, [1, 2]]).all()  # type: ignore
        assert (dst_image.mat[:, :, 0] != image.mat[:, :, 0]).all()
        write_image(f'hsl_{delta}.jpg', dst_image)

    image = read_image('Lenna.png').to_rgb_image()

    for delta in [50, 100, 150, 200]:
        dst_image = color_shift.distort_image(ColorShiftConfig(delta), image)
        ratio = (dst_image.mat == image.mat).sum() / np.prod(image.mat.shape)
        assert ratio < 0.3
        write_image(f'rgb_{delta}.jpg', dst_image)


def test_brightness_shift():
    image = read_image('Lenna.png').to_hsl_image()

    dst_image = brightness_shift.distort_image(
        BrightnessShiftConfig(delta=255),
        image,
    )
    assert (dst_image.mat[:, :, [0, 1]] == image.mat[:, :, [0, 1]]).all()  # type: ignore
    assert (dst_image.mat[:, :, 2] != image.mat[:, :, 2]).all()
    write_image('hsl.jpg', dst_image)
    dst_image = dst_image.to_rgb_image()
    assert (dst_image.mat == 255).all()

    image = read_image('Lenna.png').to_hsv_image()

    dst_image = brightness_shift.distort_image(
        BrightnessShiftConfig(delta=255),
        image,
    )
    assert (dst_image.mat[:, :, [0, 1]] == image.mat[:, :, [0, 1]]).all()  # type: ignore
    ratio = (dst_image.mat[:, :, 2] != image.mat[:, :, 2]).sum() / np.prod(image.shape)
    assert 0.99 < ratio
    write_image('hsv.jpg', dst_image)
    dst_image = dst_image.to_rgb_image()
    assert not (dst_image.mat == 255).all()


def test_std_shift():
    image = read_image('Lenna.png').to_rgb_image()
    std = np.std(image.mat.reshape(-1, 3), axis=0)
    mean = np.mean(image.mat.reshape(-1, 3), axis=0)

    dst_image = std_shift.distort_image(
        StdShiftConfig(scale=1.5),
        image,
    )
    write_image('dst.jpg', dst_image)
    dst_std = np.std(dst_image.mat.reshape(-1, 3), axis=0)
    dst_mean = np.mean(dst_image.mat.reshape(-1, 3), axis=0)

    assert (np.abs(dst_std / std - 1.5) < 0.1).all()
    assert (np.abs(mean - dst_mean) < 2).all()


def test_boundary_equalization():
    image = read_image('Unequalized.jpg').to_hsl_image()

    dst_image = boundary_equalization.distort_image(
        BoundaryEqualizationConfig(),
        image,
    )
    assert (dst_image.mat.reshape(-1, 3).min(axis=0) == 0).all()
    assert dst_image.mat.reshape(-1, 3).max(axis=0)[-1] == 255
    write_image('Unequalized.jpg', dst_image)

    image = read_image('Lenna.png').to_hsl_image()
    dst_image = boundary_equalization.distort_image(
        BoundaryEqualizationConfig(),
        image,
    )
    assert (dst_image.mat.reshape(-1, 3).min(axis=0) == 0).all()
    assert (dst_image.mat.reshape(-1, 3).max(axis=0) == 255).all()
    write_image('Lenna_hsl.jpg', dst_image)

    image = read_image('Lenna.png').to_hsv_image()
    dst_image = boundary_equalization.distort_image(
        BoundaryEqualizationConfig(),
        image,
    )
    assert (dst_image.mat.reshape(-1, 3).min(axis=0) == 0).all()
    assert (dst_image.mat.reshape(-1, 3).max(axis=0) == 255).all()
    write_image('Lenna_hsv.jpg', dst_image)


def test_histogram_equalization():
    image = read_image('Unequalized.jpg').to_hsl_image()

    dst_image = histogram_equalization.distort_image(
        HistogramEqualizationConfig(),
        image,
    )
    assert (dst_image.mat.reshape(-1, 3).min(axis=0) == 0).all()
    assert dst_image.mat.reshape(-1, 3).max(axis=0)[-1] == 255
    write_image('Unequalized.jpg', dst_image)

    image = read_image('Lenna.png').to_hsl_image()
    dst_image = histogram_equalization.distort_image(
        HistogramEqualizationConfig(),
        image,
    )
    assert (dst_image.mat.reshape(-1, 3).min(axis=0) == 0).all()
    assert (dst_image.mat.reshape(-1, 3).max(axis=0) == 255).all()
    write_image('Lenna_hsl.jpg', dst_image)

    image = read_image('Lenna.png').to_hsv_image()
    dst_image = histogram_equalization.distort_image(
        HistogramEqualizationConfig(),
        image,
    )
    assert (dst_image.mat.reshape(-1, 3).min(axis=0) == 0).all()
    assert (dst_image.mat.reshape(-1, 3).max(axis=0) == 255).all()
    write_image('Lenna_hsv.jpg', dst_image)


def test_complement():
    image = read_image('Lenna.png').to_rgb_image()

    dst_image = complement.distort_image(ComplementConfig(), image)
    assert (dst_image.mat + image.mat == 255).all()
    write_image('default.jpg', dst_image)

    threshold = 100
    mask = (image.mat < threshold)
    dst_image = complement.distort_image(ComplementConfig(threshold=threshold), image)
    assert (dst_image.mat[mask] == image.mat[mask]).all()
    write_image('threshold_default.jpg', dst_image)

    mask = (image.mat > threshold)
    dst_image = complement.distort_image(
        ComplementConfig(threshold=threshold, enable_threshold_lte=True),
        image,
    )
    assert (dst_image.mat[mask] == image.mat[mask]).all()
    write_image('threshold_lte.jpg', dst_image)


def test_posterization():
    image = read_image('Lenna.png').to_rgb_image()

    for num_bits in range(1, 8):
        dst_image = posterization.distort_image(
            PosterizationConfig(num_bits=num_bits),
            image,
        )
        assert (np.bitwise_and(dst_image.mat, 0x1 << (num_bits - 1)) == 0).all()
        write_image(f'num_bits_{num_bits}.jpg', dst_image)


def test_color_balance():
    image = read_image('Lenna.png').to_rgb_image()

    dst_image = color_balance.distort_image(ColorBalanceConfig(ratio=0.0), image)
    assert (dst_image.to_hsv_image().mat[:, :, 1] == 0).all()
    write_image('rgb_0.jpg', dst_image)

    dst_image = color_balance.distort_image(ColorBalanceConfig(ratio=0.5), image)
    write_image('rgb_0.5.jpg', dst_image)

    dst_image = color_balance.distort_image(ColorBalanceConfig(ratio=1.0), image)
    write_image('rgb_1.jpg', dst_image)

    image = read_image('Lenna.png').to_hsl_image()

    dst_image = color_balance.distort_image(ColorBalanceConfig(ratio=0.0), image)
    write_image('hsl_0.jpg', dst_image)

    dst_image = color_balance.distort_image(ColorBalanceConfig(ratio=0.5), image)
    write_image('hsl_0.5.jpg', dst_image)

    dst_image = color_balance.distort_image(ColorBalanceConfig(ratio=1.0), image)
    write_image('hsl_1.jpg', dst_image)


def test_channel_permutate():
    image = read_image('Lenna.png').to_rgb_image()

    # [2, 0, 1]
    rng = default_rng(0)
    dst_image = channel_permutation.distort_image(
        ChannelPermutationConfig(),
        image,
        rng=rng,
    )

    assert (dst_image.mat[:, :, 0] == image.mat[:, :, 2]).all()
    assert (dst_image.mat[:, :, 1] == image.mat[:, :, 0]).all()
    assert (dst_image.mat[:, :, 2] == image.mat[:, :, 1]).all()
    write_image('102.jpg', dst_image)


def test_gaussian_blur():
    image = read_image('Lenna.png').to_rgb_image()

    dst_image = gaussian_blur.distort_image(
        GaussianBlurConfig(sigma=2),
        image,
    )
    assert dst_image.shape == image.shape
    write_image('rgb_2.jpg', dst_image)

    image = read_image('Lenna.png').to_hsl_image()

    dst_image = gaussian_blur.distort_image(
        GaussianBlurConfig(sigma=2),
        image,
    )
    assert dst_image.shape == image.shape
    write_image('hsl_2.jpg', dst_image)


def test_defocus_blur():
    image = read_image('Lenna.png').to_rgb_image()

    for radius in [2, 3, 4, 10, 20, 30]:
        dst_image = defocus_blur.distort_image(
            DefocusBlurConfig(radius=radius),
            image,
        )
        write_image(f'r{radius}.jpg', dst_image)


def test_motion_blur():
    image = read_image('Lenna.png').to_rgb_image()

    for radius in [5, 10, 15, 20]:
        for angle in [-45, 0, 45]:
            dst_image = motion_blur.distort_image(
                MotionBlurConfig(radius=radius, angle=angle),
                image,
            )
            write_image(f'r{radius}_a{angle}.jpg', dst_image)


def test_glass_blur():
    image = read_image('Lenna.png').to_rgb_image()
    rng = default_rng(3)

    for sigma in [0.5, 1, 2]:
        for loop in [3, 5, 10, 20]:
            dst_image = glass_blur.distort_image(
                GlassBlurConfig(sigma=sigma, loop=loop),
                image,
                rng=rng,
            )
            write_image(f's{sigma}_l{loop}.jpg', dst_image)


def test_zoom_in_blur():
    image = read_image('Lenna.png').to_rgb_image()

    for ratio in [0.1, 0.15, 0.2]:
        dst_image = zoom_in_blur.distort_image(
            ZoomInBlurConfig(ratio=ratio),
            image,
        )
        write_image(f'r{ratio}.jpg', dst_image)


def test_jpeg_quality():
    image = read_image('Lenna.png').to_rgb_image()

    for quality in [95, 80, 50, 20]:
        dst_image = jpeg_quality.distort_image(JpegQualityConfig(quality=quality), image)
        write_image(f'{quality}.jpg', dst_image)


def test_pixelation():
    image = read_image('Lenna.png').to_rgb_image()

    for ratio in [0.9, 0.5, 0.1]:
        dst_image = pixelation.distort_image(PixelationConfig(ratio=ratio), image)
        write_image(f'{ratio}.jpg', dst_image)


def test_generate_diamond_square_mask():
    from vkit.element import Image
    from vkit.mechanism.distortion.photometric.effect import generate_diamond_square_mask

    for roughness in [1.0, 0.9, 0.8, 0.5, 0.2, 0.0]:
        rng = default_rng(3)
        mask = generate_diamond_square_mask((400, 400), roughness, rng)
        mask = np.round(mask * 255).astype(np.uint8)
        write_image(f'{roughness}.jpg', Image(mat=mask))


def test_fog():
    image = read_image('Lenna.png').to_rgb_image()

    for roughness in [0.8, 0.7, 0.6, 0.5, 0.2]:
        rng = default_rng(3)
        dst_image = fog.distort_image(FogConfig(roughness=roughness), image, rng=rng)
        write_image(f'{roughness}.jpg', dst_image)

    image = read_image('Lenna.png').to_grayscale_image()
    rng = default_rng(3)
    dst_image = fog.distort_image(FogConfig(roughness=0.5), image, rng=rng)
    write_image('grayscalae.jpg', dst_image)


def test_line_streak():
    image = read_image('Lenna.png').to_rgb_image()

    dst_image = line_streak.distort_image({}, image)
    write_image('default.jpg', dst_image)

    dst_image = line_streak.distort_image({'alpha': 0.2}, image)
    write_image('alpha-0.2.jpg', dst_image)

    dst_image = line_streak.distort_image({'thickness': 10, 'gap': 10}, image)
    write_image('thickness-10-gap-10.jpg', dst_image)

    dst_image = line_streak.distort_image(
        {
            'thickness': 2,
            'gap': 50,
            'dash_thickness': 10,
            'dash_gap': 5,
        },
        image,
    )
    write_image('dash.jpg', dst_image)

    dst_image = line_streak.distort_image({'enable_hori': False}, image)
    write_image('no-hori.jpg', dst_image)

    dst_image = line_streak.distort_image({'enable_vert': False}, image)
    write_image('no-vert.jpg', dst_image)


def test_rectangle_streak():
    image = read_image('Lenna.png').to_rgb_image()

    dst_image = rectangle_streak.distort_image({}, image)
    write_image('default.jpg', dst_image)

    dst_image = rectangle_streak.distort_image({'thickness': 3}, image)
    write_image('thickness-3.jpg', dst_image)

    dst_image = rectangle_streak.distort_image({'dash_thickness': 10, 'dash_gap': 5}, image)
    write_image('dash.jpg', dst_image)

    dst_image = rectangle_streak.distort_image({'short_side_min': 5}, image)
    write_image('short_side_min-5.jpg', dst_image)

    dst_image = rectangle_streak.distort_image({'aspect_ratio': 0.5}, image)
    write_image('aspect_ratio-0.5.jpg', dst_image)

    dst_image = rectangle_streak.distort_image({'aspect_ratio': 2}, image)
    write_image('aspect_ratio-2.jpg', dst_image)


def test_ellipse_streak():
    image = read_image('Lenna.png').to_rgb_image()

    dst_image = ellipse_streak.distort_image({}, image)
    write_image('default.jpg', dst_image)

    dst_image = ellipse_streak.distort_image({'thickness': 2}, image)
    write_image('thickness-2.jpg', dst_image)

    dst_image = ellipse_streak.distort_image({'aspect_ratio': 0.5}, image)
    write_image('aspect_ratio-0.5.jpg', dst_image)

    dst_image = ellipse_streak.distort_image({'aspect_ratio': 2}, image)
    write_image('aspect_ratio-2.jpg', dst_image)
