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
from vkit.engine.distortion import *
from tests.opt import read_image, write_image


def test_rotate():
    image = read_image('Lenna.png').to_rgb_image().to_resized_image(567, 440)
    write_image('567-440.jpg', image)
    dst_image = rotate.distort_image({'angle': 132}, image)
    write_image('angle-132.jpg', dst_image)
