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
from vkit.element import Mask
from vkit.mechanism.painter import Painter
import cv2 as cv
from tests.opt import read_image, write_image


def test_to_bounding_rectangular_polygon():
    cheems_image = read_image('Cheems.png').to_rgb_image()
    cheems_image = cheems_image.to_resized_image(resized_height=100, resized_width=100)
    cheems_mask = Mask(mat=cv.Canny(cheems_image.mat, 100, 200))
    polygon = cheems_mask.to_external_polygon()
    polygon = polygon.to_shifted_polygon(offset_y=100, offset_x=100)
    shape = (300, 300)
    for angle in range(0, 360, 5):
        bounding_rectangular_polygon = polygon.to_bounding_rectangular_polygon(shape, angle=angle)
        painter = Painter.create(shape)
        painter.paint_polygons([bounding_rectangular_polygon, polygon])
        write_image(f'{angle}.jpg', painter.image)
