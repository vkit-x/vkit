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
from typing import cast, Union, Tuple, Sequence, Iterable, Any, Optional

import cv2 as cv
import numpy as np
from PIL import ImageColor as PilImageColor

from vkit.utility import PathType
from vkit.element import (
    Shapable,
    Point,
    PointList,
    Line,
    Box,
    Polygon,
    Mask,
    ScoreMap,
    Image,
    ImageMode,
)


class Painter:

    # https://mokole.com/palette.html
    PALETTE = (
        # darkgreen
        '#006400',
        # darkblue
        '#00008b',
        # maroon3
        '#b03060',
        # red
        '#ff0000',
        # yellow
        '#ffff00',
        # burlywood
        '#deb887',
        # lime
        '#00ff00',
        # aqua
        '#00ffff',
        # fuchsia
        '#ff00ff',
        # cornflower
        '#6495ed',
    )

    @classmethod
    def get_rgb_tuple_from_color_name(cls, color_name: str) -> Tuple[int, int, int]:
        # https://pillow.readthedocs.io/en/stable/reference/ImageColor.html#color-names
        return PilImageColor.getrgb(color_name)  # type: ignore

    @classmethod
    def get_complementary_rgba_tuple(
        cls, rgba_tuple: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        return tuple(255 - val if idx < 3 else val for idx, val in enumerate(rgba_tuple))

    @classmethod
    def get_color_names(
        cls,
        elements_or_num_elements: Union[Iterable[Any], int],
        palette: Sequence[str] = PALETTE,
    ):
        if isinstance(elements_or_num_elements, int):
            elements = range(elements_or_num_elements)
        else:
            elements = elements_or_num_elements
        return tuple(palette[idx % len(palette)] for idx, _ in enumerate(elements))

    @classmethod
    def get_rgb_tuples(
        cls,
        elements_or_num_elements: Union[Iterable[Any], int],
        palette: Sequence[str] = PALETTE,
    ):
        color_names = cls.get_color_names(elements_or_num_elements, palette=palette)
        return tuple(cls.get_rgb_tuple_from_color_name(color_name) for color_name in color_names)

    @classmethod
    def get_rgba_tuples_from_color_names(
        cls,
        num_elements: int,
        color: Optional[Union[str, Iterable[str], Iterable[int]]],
        alpha: float,
        palette: Sequence[str] = PALETTE,
    ):
        if color is None:
            rgb_tuples = cls.get_rgb_tuples(num_elements, palette=palette)

        elif isinstance(color, str):
            rgb_tuple = cls.get_rgb_tuple_from_color_name(color)
            rgb_tuples = (rgb_tuple,) * num_elements

        else:
            colors = tuple(color)
            if not colors:
                color_names = ()

            elif isinstance(colors[0], str):
                color_names = cast(Tuple[str], colors)

            elif isinstance(colors[0], int):
                color_indices = cast(Tuple[int], colors)
                color_names = [palette[color_idx % len(palette)] for color_idx in color_indices]

            else:
                raise NotImplementedError()

            rgb_tuples = tuple(
                cls.get_rgb_tuple_from_color_name(color_name) for color_name in color_names
            )

        alpha_uint8 = round(255 * alpha)
        return tuple((*rgb_tuple, alpha_uint8) for rgb_tuple in rgb_tuples)

    @classmethod
    def create(
        cls,
        image_or_shapable_or_shape: Union[Image, Shapable, Tuple[int, int]],
        num_channels: int = 3,
        value: Union[Tuple[int, ...], int] = 255,
    ):
        if isinstance(image_or_shapable_or_shape, Image):
            image = image_or_shapable_or_shape
            image = image.copy()

        else:
            if isinstance(image_or_shapable_or_shape, Shapable):
                shape = image_or_shapable_or_shape.shape
            else:
                shape = image_or_shapable_or_shape

            image = Image.from_shape(
                shape=shape,
                num_channels=num_channels,
                value=value,
            )

        return Painter(image)

    def __init__(self, image: Image):
        self.image = image.copy()

    def copy(self):
        return Painter(image=self.image.copy())

    def generate_layer_image(self):
        # RGBA.
        return Image.from_shapable(
            self.image,
            num_channels=4,
            value=0,
        )

    def overlay_layer_image(self, layer_image: Image):
        alpha = layer_image.mat[:, :, 3].astype(np.float32) / 255.0

        layer_image = Image(mat=layer_image.mat[:, :, :3], mode=ImageMode.RGB)
        layer_image = layer_image.to_target_mode_image(self.image.mode)

        Box.from_shapable(layer_image).fill_image(
            self.image,
            value=layer_image,
            alpha=alpha,
        )

    @classmethod
    def paint_text_to_layer_image(
        cls,
        layer_image: Image,
        text: str,
        point: Point,
        rgba_tuple: Tuple[int, int, int, int],
        font_scale: float = 1.0,
        enable_complementary_rgba: bool = True,
    ):
        if enable_complementary_rgba:
            color = cls.get_complementary_rgba_tuple(rgba_tuple)
        else:
            color = rgba_tuple

        cv.putText(
            layer_image.mat,
            text=text,
            org=(point.x, point.y),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=cv.LINE_AA,
        )

    def paint_texts(
        self,
        texts: Iterable[str],
        points: Iterable[Point],
        color: Optional[Union[str, Iterable[str], Iterable[int]]] = None,
        alpha: float = 0.5,
        palette: Sequence[str] = PALETTE,
    ):
        layer_image = self.generate_layer_image()

        texts = tuple(texts)
        points = tuple(points)
        assert len(texts) == len(points)
        rgba_tuples = self.get_rgba_tuples_from_color_names(
            len(points),
            color,
            alpha,
            palette=palette,
        )
        for text, point, rgba_tuple in zip(texts, points, rgba_tuples):
            self.paint_text_to_layer_image(
                layer_image=layer_image,
                text=text,
                point=point,
                rgba_tuple=rgba_tuple,
            )

        self.overlay_layer_image(layer_image)

    def paint_points(
        self,
        points: Union[PointList, Iterable[Point]],
        radius: int = 1,
        enable_index: bool = False,
        color: Optional[Union[str, Iterable[str], Iterable[int]]] = None,
        alpha: float = 0.5,
        palette: Sequence[str] = PALETTE,
    ):
        layer_image = self.generate_layer_image()

        if not isinstance(points, PointList):
            points = PointList(points)

        rgba_tuples = self.get_rgba_tuples_from_color_names(
            len(points),
            color,
            alpha,
            palette=palette,
        )
        for idx, (point, rgba_tuple) in enumerate(zip(points, rgba_tuples)):
            cv.circle(
                layer_image.mat,
                center=(point.x, point.y),
                radius=radius,
                color=rgba_tuple,
                thickness=cv.FILLED,
                lineType=cv.LINE_AA,
            )

            if enable_index:
                self.paint_text_to_layer_image(
                    layer_image=layer_image,
                    text=str(idx),
                    point=point,
                    rgba_tuple=rgba_tuple,
                )

        self.overlay_layer_image(layer_image)

    def paint_lines(
        self,
        lines: Iterable[Line],
        thickness: int = 1,
        enable_arrow: bool = False,
        arrow_length_ratio: float = 0.1,
        enable_index: bool = False,
        color: Optional[Union[str, Iterable[str], Iterable[int]]] = None,
        alpha: float = 0.5,
        palette: Sequence[str] = PALETTE,
    ):
        layer_image = self.generate_layer_image()

        lines = tuple(lines)
        rgba_tuples = self.get_rgba_tuples_from_color_names(
            len(lines),
            color,
            alpha,
            palette=palette,
        )
        for idx, (line, rgba_tuple) in enumerate(zip(lines, rgba_tuples)):
            if not enable_arrow:
                cv.line(
                    layer_image.mat,
                    pt1=(line.point_begin.x, line.point_begin.y),
                    pt2=(line.point_end.x, line.point_end.y),
                    color=rgba_tuple,
                    thickness=thickness,
                    lineType=cv.LINE_AA,
                )
            else:
                cv.arrowedLine(
                    layer_image.mat,
                    pt1=(line.point_begin.x, line.point_begin.y),
                    pt2=(line.point_end.x, line.point_end.y),
                    color=rgba_tuple,
                    thickness=thickness,
                    line_type=cv.LINE_AA,
                    tipLength=arrow_length_ratio,
                )

            if enable_index:
                center_point = line.get_center_point()
                self.paint_text_to_layer_image(
                    layer_image=layer_image,
                    text=str(idx),
                    point=center_point,
                    rgba_tuple=rgba_tuple,
                )

        self.overlay_layer_image(layer_image)

    def paint_boxes(
        self,
        boxes: Iterable[Box],
        enable_index: bool = False,
        color: Optional[Union[str, Iterable[str], Iterable[int]]] = None,
        border_thickness: Optional[int] = None,
        alpha: float = 0.5,
        palette: Sequence[str] = PALETTE,
    ):
        layer_image = self.generate_layer_image()

        boxes = tuple(boxes)
        rgba_tuples = self.get_rgba_tuples_from_color_names(
            len(boxes),
            color,
            alpha,
            palette=palette,
        )

        for idx, (box, rgba_tuple) in enumerate(zip(boxes, rgba_tuples)):
            box.fill_image(image=layer_image, value=rgba_tuple)
            if border_thickness:
                inner_box = Box(
                    up=box.up + border_thickness,
                    down=box.down - border_thickness,
                    left=box.left + border_thickness,
                    right=box.right - border_thickness,
                )
                if inner_box.valid:
                    inner_box.fill_image(image=layer_image, value=0)

            if enable_index:
                center_point = box.get_center_point()
                self.paint_text_to_layer_image(
                    layer_image=layer_image,
                    text=str(idx),
                    point=center_point,
                    rgba_tuple=rgba_tuple,
                    enable_complementary_rgba=(border_thickness is None),
                )

        self.overlay_layer_image(layer_image)

    def paint_polygons(
        self,
        polygons: Iterable[Polygon],
        color: Optional[Union[str, Iterable[str], Iterable[int]]] = None,
        alpha: float = 0.5,
        palette: Sequence[str] = PALETTE,
        enable_index: bool = False,
        enable_polygon_points: bool = False,
        polygon_points_color: str = 'red',
        polygon_points_alpha: float = 1.0,
    ):
        layer_image = self.generate_layer_image()

        polygons = tuple(polygons)
        rgba_tuples = self.get_rgba_tuples_from_color_names(
            len(polygons),
            color,
            alpha,
            palette=palette,
        )
        for idx, (polygon, rgba_tuple) in enumerate(zip(polygons, rgba_tuples)):
            polygon.fill_image(image=layer_image, value=rgba_tuple)

        if enable_index:
            for idx, (polygon, rgba_tuple) in enumerate(zip(polygons, rgba_tuples)):
                center_point = polygon.get_center_point()
                self.paint_text_to_layer_image(
                    layer_image=layer_image,
                    text=str(idx),
                    point=center_point,
                    rgba_tuple=rgba_tuple,
                )

        if enable_polygon_points:
            for idx, (polygon, rgba_tuple) in enumerate(zip(polygons, rgba_tuples)):
                self.paint_points(
                    polygon.points,
                    color=polygon_points_color,
                    alpha=polygon_points_alpha,
                )

        self.overlay_layer_image(layer_image)

    def paint_mask(
        self,
        mask: Mask,
        color: str = 'red',
        alpha: float = 0.5,
    ):
        layer_image = self.generate_layer_image()

        rgb_tuple = self.get_rgb_tuple_from_color_name(color)
        alpha_uint8 = round(255 * alpha)
        mask.fill_image(image=layer_image, value=(*rgb_tuple, alpha_uint8))

        self.overlay_layer_image(layer_image)

    def paint_masks(
        self,
        masks: Iterable[Mask],
        color: Optional[Union[str, Iterable[str], Iterable[int]]] = None,
        alpha: float = 0.5,
        palette: Sequence[str] = PALETTE,
    ):
        masks = tuple(masks)
        rgba_tuples = self.get_rgba_tuples_from_color_names(
            len(masks),
            color,
            alpha,
            palette=palette,
        )

        layer_image = self.generate_layer_image()
        layer_image.fill_by_mask_value_tuples(zip(masks, rgba_tuples))
        self.overlay_layer_image(layer_image)

    def paint_score_map(
        self,
        score_map: ScoreMap,
        enable_boundary_equalization: bool = False,
        enable_center_shift: bool = False,
        cv_colormap: int = cv.COLORMAP_JET,
        alpha: float = 0.5,
    ):
        layer_image = self.generate_layer_image()

        mat = score_map.mat.copy()

        if score_map.is_prob:
            mat *= 255.0

        if enable_boundary_equalization:
            # Equalize to [0, 255]
            val_min = np.min(mat)
            mat -= val_min
            val_max = np.max(mat)
            mat *= 255.0
            mat /= val_max

        elif enable_center_shift:
            mat *= 127.5

        mat = np.clip(mat, 0, 255).astype(np.uint8)

        # Apply color map.
        color_mat = cv.applyColorMap(mat, cv_colormap)
        color_mat = cv.cvtColor(color_mat, cv.COLOR_BGR2RGB)

        # Add alpha channel.
        alpha_uint8 = round(255 * alpha)
        color_mat = np.dstack((
            color_mat,
            np.full(color_mat.shape[:2], alpha_uint8, dtype=np.uint8),
        ))

        if score_map.box:
            score_map.box.fill_image(layer_image, color_mat)
        else:
            layer_image.assign_mat(color_mat)

        self.overlay_layer_image(layer_image)

    def to_file(self, path: PathType, disable_to_rgb_image: bool = False):
        self.image.to_file(path=path, disable_to_rgb_image=disable_to_rgb_image)
