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
##########################
# Photometric Distortion #
##########################
from .photometric.opt import OutOfBoundBehavior
from .photometric.color import (
    MeanShiftConfig,
    mean_shift,
    ColorShiftConfig,
    color_shift,
    BrightnessShiftConfig,
    brightness_shift,
    StdShiftConfig,
    std_shift,
    BoundaryEqualizationConfig,
    boundary_equalization,
    HistogramEqualizationConfig,
    histogram_equalization,
    ComplementConfig,
    complement,
    PosterizationConfig,
    posterization,
    ColorBalanceConfig,
    color_balance,
    ChannelPermutationConfig,
    channel_permutation,
)
from .photometric.blur import (
    GaussianBlurConfig,
    gaussian_blur,
    DefocusBlurConfig,
    defocus_blur,
    MotionBlurConfig,
    motion_blur,
    GlassBlurConfig,
    glass_blur,
    ZoomInBlurConfig,
    zoom_in_blur,
)
from .photometric.noise import (
    GaussionNoiseConfig,
    gaussion_noise,
    PoissonNoiseConfig,
    poisson_noise,
    ImpulseNoiseConfig,
    impulse_noise,
    SpeckleNoiseConfig,
    speckle_noise,
)
from .photometric.effect import (
    JpegQualityConfig,
    jpeg_quality,
    PixelationConfig,
    pixelation,
    FogConfig,
    fog,
)
from .photometric.streak import (
    LineStreakConfig,
    line_streak,
    RectangleStreakConfig,
    rectangle_streak,
    EllipseStreakConfig,
    ellipse_streak,
)

########################
# Geometric Distortion #
########################
from .geometric.affine import (
    ShearHoriConfig,
    shear_hori,
    ShearVertConfig,
    shear_vert,
    RotateConfig,
    rotate,
    SkewHoriConfig,
    skew_hori,
    SkewVertConfig,
    skew_vert,
)
from .geometric.mls import (
    SimilarityMlsConfig,
    similarity_mls,
)
from .geometric.camera import (
    CameraModelConfig,
    CameraPlaneOnlyConfig,
    camera_plane_only,
    CameraCubicCurveConfig,
    camera_cubic_curve,
    CameraPlaneLineFoldConfig,
    camera_plane_line_fold,
    CameraPlaneLineCurveConfig,
    camera_plane_line_curve,
)
