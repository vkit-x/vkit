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
import string
import cv2 as cv
from numpy.random import default_rng


def debug_collect_cv_qrcode_size():
    rng = default_rng(42)
    ascii_letters = tuple(string.ascii_letters)

    qrcode_encoder = cv.QRCodeEncoder.create()
    # NOTE: 2955 fails. length = 2954 -> shape = 181
    length = 2954
    mat = qrcode_encoder.encode(''.join(rng.choice(ascii_letters) for _ in range(length)))
    print(f'length={length}, shape={mat.shape}')
