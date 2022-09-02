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
from vkit.engine.distortion.geometric.camera import *
from tests.opt import read_image, write_image


def test_camera_model():

    image = read_image('Lenna.png').to_rgb_image()

    for prefix, rotation_unit_vec in [
        ('x', [1.0, 0.0, 0.0]),
        ('y', [0.0, 1.0, 0.0]),
        ('z', [0.0, 0.0, 1.0]),
        ('xy', [1.0, 1.0, 0.0]),
    ]:
        nop_image = camera_plane_only.distort_image(
            CameraPlaneOnlyConfig(
                camera_model_config=CameraModelConfig(
                    rotation_unit_vec=rotation_unit_vec,
                    rotation_theta=0,
                ),
                grid_size=50,
            ),
            image,
        )
        assert nop_image.shape == image.shape

        for rotation_theta in range(-45, 46, 5):
            write_image(
                f'{prefix}-{"pos" if rotation_theta >= 0 else "neg"}-{abs(rotation_theta)}.jpg',
                camera_plane_only.distort_image(
                    CameraPlaneOnlyConfig(
                        camera_model_config=CameraModelConfig(
                            rotation_unit_vec=rotation_unit_vec,
                            rotation_theta=rotation_theta,
                        ),
                        grid_size=50,
                    ),
                    image,
                ),
            )
