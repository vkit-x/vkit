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
