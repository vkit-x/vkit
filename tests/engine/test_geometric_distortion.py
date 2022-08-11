from vkit.engine.distortion import *
from tests.opt import read_image, write_image


def test_rotate():
    image = read_image('Lenna.png').to_rgb_image().to_resized_image(567, 440)
    write_image('567-440.jpg', image)
    dst_image = rotate.distort_image({'angle': 132}, image)
    write_image('angle-132.jpg', dst_image)
