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
from typing import (
    cast,
    Sequence,
    List,
    Optional,
    Mapping,
    Dict,
    DefaultDict,
    Iterable,
    Set,
    Tuple,
    Union,
)
from enum import Enum, unique
from collections import defaultdict

import attrs
import iolite as io
import numpy as np
import cv2 as cv

from vkit.utility import (
    attrs_lazy_field,
    get_cattrs_converter_ignoring_init_equals_false,
    dyn_structure,
    PathType,
)
from vkit.element import (
    Shapable,
    Point,
    PointList,
    Box,
    Polygon,
    Mask,
    ScoreMap,
    Image,
)


@attrs.define(frozen=True)
class FontGlyphInfo:
    tags: Sequence[str]
    ascent_plus_pad_up_min_to_font_size_ratio: float
    height_min_to_font_size_ratio: float
    width_min_to_font_size_ratio: float


@attrs.define
class FontGlyphInfoCollection:
    font_glyph_infos: Sequence[FontGlyphInfo]

    _tag_to_font_glyph_info: Mapping[str, FontGlyphInfo] = attrs_lazy_field()

    def lazy_post_init_tag_to_font_glyph_info(self):
        if self._tag_to_font_glyph_info:
            return self._tag_to_font_glyph_info

        tag_to_font_glyph_info = {}
        for font_glyph_info in self.font_glyph_infos:
            assert font_glyph_info.tags
            for tag in font_glyph_info.tags:
                assert tag not in tag_to_font_glyph_info
                tag_to_font_glyph_info[tag] = font_glyph_info

        self._tag_to_font_glyph_info = cast(Mapping[str, FontGlyphInfo], tag_to_font_glyph_info)
        return self._tag_to_font_glyph_info

    @property
    def tag_to_font_glyph_info(self):
        return self.lazy_post_init_tag_to_font_glyph_info()


@attrs.define
class FontVariant:
    char_to_tags: Mapping[str, Sequence[str]]
    font_file: PathType
    font_glyph_info_collection: FontGlyphInfoCollection
    is_ttc: bool = False
    ttc_font_index: Optional[int] = None


@unique
class FontMode(Enum):
    # ttc file.
    TTC = 'ttc'
    # Grouped ttf file(s).
    VTTC = 'vttc'
    # Grouped otf file(s).
    VOTC = 'votc'


@attrs.define
class FontMeta:
    name: str
    mode: FontMode
    char_to_tags: Mapping[str, Sequence[str]]
    font_files: Sequence[str]
    font_glyph_info_collection: FontGlyphInfoCollection
    # NOTE: ttc_font_index_max is inclusive.
    ttc_font_index_max: Optional[int] = None

    _chars: Sequence[str] = attrs_lazy_field()

    def lazy_post_init_chars(self):
        if self._chars:
            return self._chars

        self._chars = cast(Sequence[str], sorted(self.char_to_tags))
        return self._chars

    @property
    def chars(self):
        return self.lazy_post_init_chars()

    def __repr__(self):
        return (
            'FontMeta('
            f'name="{self.name}", '
            f'mode={self.mode}, '
            f'num_chars={len(self.char_to_tags)}), '
            f'font_files={self.font_files}, '
            f'ttc_font_index_max={self.ttc_font_index_max})'
        )

    @classmethod
    def from_file(
        cls,
        path: PathType,
        font_file_prefix: Optional[PathType] = None,
    ):
        font = dyn_structure(path, FontMeta, force_path_type=True)

        if font_file_prefix:
            font_file_prefix_fd = io.folder(font_file_prefix, exists=True)
            font_files = []
            for font_file in font.font_files:
                font_file = str(io.file(font_file_prefix_fd / io.file(font_file), exists=True))
                font_files.append(font_file)
            font = attrs.evolve(font, font_files=font_files)

        return font

    def to_file(
        self,
        path: PathType,
        font_file_prefix: Optional[PathType] = None,
    ):
        font = self

        if font_file_prefix:
            font_file_prefix_fd = io.folder(font_file_prefix)
            font_files = []
            for font_file in self.font_files:
                font_files.append(str(io.file(font_file).relative_to(font_file_prefix_fd)))
            font = attrs.evolve(self, font_files=font_files)

        converter = get_cattrs_converter_ignoring_init_equals_false()
        io.write_json(path, converter.unstructure(font), indent=2, ensure_ascii=False)

    @property
    def num_font_variants(self):
        if self.mode in (FontMode.VOTC, FontMode.VTTC):
            return len(self.font_files)

        elif self.mode == FontMode.TTC:
            assert self.ttc_font_index_max is not None
            return self.ttc_font_index_max + 1

        else:
            raise NotImplementedError()

    def get_font_variant(self, variant_idx: int):
        if self.mode in (FontMode.VOTC, FontMode.VTTC):
            assert variant_idx < len(self.font_files)
            return FontVariant(
                char_to_tags=self.char_to_tags,
                font_file=io.file(self.font_files[variant_idx]),
                font_glyph_info_collection=self.font_glyph_info_collection,
            )

        elif self.mode == FontMode.TTC:
            assert self.ttc_font_index_max is not None
            assert variant_idx <= self.ttc_font_index_max
            return FontVariant(
                char_to_tags=self.char_to_tags,
                font_file=io.file(self.font_files[0]),
                font_glyph_info_collection=self.font_glyph_info_collection,
                is_ttc=True,
                ttc_font_index=variant_idx,
            )

        else:
            raise NotImplementedError()


class FontCollectionFolderTree:
    FONT = 'font'
    FONT_META = 'font_meta'


@attrs.define
class FontCollection:
    font_metas: Sequence[FontMeta]

    _name_to_font_meta: Optional[Mapping[str, FontMeta]] = attrs_lazy_field()
    _char_to_font_meta_names: Optional[Mapping[str, Set[str]]] = attrs_lazy_field()

    def lazy_post_init(self):
        initialized = (self._name_to_font_meta is not None)
        if initialized:
            return

        name_to_font_meta: Dict[str, FontMeta] = {}
        char_to_font_meta_names: DefaultDict[str, Set[str]] = defaultdict(set)
        for font_meta in self.font_metas:
            assert font_meta.name not in name_to_font_meta
            name_to_font_meta[font_meta.name] = font_meta
            for char in font_meta.chars:
                char_to_font_meta_names[char].add(font_meta.name)
        self._name_to_font_meta = name_to_font_meta
        self._char_to_font_meta_names = dict(char_to_font_meta_names)

    @property
    def name_to_font_meta(self):
        self.lazy_post_init()
        assert self._name_to_font_meta is not None
        return self._name_to_font_meta

    @property
    def char_to_font_meta_names(self):
        self.lazy_post_init()
        assert self._char_to_font_meta_names is not None
        return self._char_to_font_meta_names

    def filter_font_metas(self, chars: Iterable[str]):
        font_meta_names = set.intersection(
            *[self.char_to_font_meta_names[char] for char in chars if not char.isspace()]
        )
        font_meta_names = sorted(font_meta_names)
        return [self.name_to_font_meta[font_meta_name] for font_meta_name in font_meta_names]

    @classmethod
    def from_folder(cls, folder: PathType):
        in_fd = io.folder(folder, expandvars=True, exists=True)
        font_fd = io.folder(in_fd / FontCollectionFolderTree.FONT, exists=True)
        font_meta_fd = io.folder(in_fd / FontCollectionFolderTree.FONT_META, exists=True)

        font_metas: List[FontMeta] = []
        for font_meta_json in font_meta_fd.glob('*.json'):
            font_metas.append(FontMeta.from_file(font_meta_json, font_fd))

        return cls(font_metas=font_metas)


@attrs.define
class FontEngineRunConfigStyle:
    # Font size.
    font_size_ratio: float = 1.0
    font_size_min: int = 12
    font_size_max: int = 96

    # Space between chars.
    char_space_min: float = 0.0
    char_space_max: float = 0.2
    char_space_mean: float = 0.1
    char_space_std: float = 0.03

    # Space between words.
    word_space_min: float = 0.3
    word_space_max: float = 1.0
    word_space_mean: float = 0.6
    word_space_std: float = 0.1

    # Effect.
    glyph_color: Tuple[int, int, int] = (0, 0, 0)
    # https://en.wikipedia.org/wiki/Gamma_correction
    glyph_color_gamma: float = 1.0

    # Implementation related options.
    freetype_force_autohint: bool = False


@unique
class FontEngineRunConfigGlyphSequence(Enum):
    HORI_DEFAULT = 'hori_default'
    VERT_DEFAULT = 'vert_default'


@attrs.define
class FontEngineRunConfig:
    height: int
    width: int
    chars: Sequence[str]
    font_variant: FontVariant

    # Sequence mode.
    glyph_sequence: FontEngineRunConfigGlyphSequence = \
        FontEngineRunConfigGlyphSequence.HORI_DEFAULT

    style: FontEngineRunConfigStyle = attrs.field(factory=FontEngineRunConfigStyle)

    # For debugging.
    return_font_variant: bool = False


@attrs.define(frozen=True)
class CharBox(Shapable):
    char: str
    box: Box

    def __attrs_post_init__(self):
        assert len(self.char) == 1 and not self.char.isspace()

    ############
    # Property #
    ############
    @property
    def up(self):
        return self.box.up

    @property
    def down(self):
        return self.box.down

    @property
    def left(self):
        return self.box.left

    @property
    def right(self):
        return self.box.right

    @property
    def height(self):
        return self.box.height

    @property
    def width(self):
        return self.box.width

    ############
    # Operator #
    ############
    def to_conducted_resized_char_box(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return attrs.evolve(
            self,
            box=self.box.to_conducted_resized_box(
                shapable_or_shape=shapable_or_shape,
                resized_height=resized_height,
                resized_width=resized_width,
            ),
        )

    def to_resized_char_box(
        self,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return attrs.evolve(
            self,
            box=self.box.to_resized_box(
                resized_height=resized_height,
                resized_width=resized_width,
            ),
        )

    def to_shifted_char_box(self, offset_y: int = 0, offset_x: int = 0):
        return attrs.evolve(
            self,
            box=self.box.to_shifted_box(offset_y=offset_y, offset_x=offset_x),
        )


@attrs.define
class CharGlyph:
    char: str
    image: Image
    score_map: Optional[ScoreMap]
    # Load from font face. See build_char_glyph.
    ascent: int
    pad_up: int
    pad_down: int
    pad_left: int
    pad_right: int
    # For rendering text line and generating char-level polygon, based on the reference char.
    ref_ascent_plus_pad_up: int
    ref_char_height: int
    ref_char_width: int

    def __attrs_post_init__(self):
        # NOTE: ascent could be negative for char like '_'.
        assert self.pad_up >= 0
        assert self.pad_down >= 0
        assert self.pad_left >= 0
        assert self.pad_right >= 0

    @property
    def height(self):
        return self.image.height

    @property
    def width(self):
        return self.image.width

    def get_glyph_mask(
        self,
        box: Optional[Box] = None,
        enable_resize: bool = False,
        cv_resize_interpolation: int = cv.INTER_CUBIC,
    ):
        if self.image.mat.ndim == 2:
            # Default or monochrome.
            np_mask = (self.image.mat > 0)

        elif self.image.mat.ndim == 3:
            # LCD.
            np_mask = np.any((self.image.mat > 0), axis=2)

        else:
            raise NotImplementedError()

        mask = Mask(mat=np_mask.astype(np.uint8))
        if box:
            if mask.shape != box.shape:
                assert enable_resize
                mask = mask.to_resized_mask(
                    resized_height=box.height,
                    resized_width=box.width,
                    cv_resize_interpolation=cv_resize_interpolation,
                )
            mask = mask.to_box_attached(box)

        return mask


@attrs.define
class TextLine:
    image: Image
    mask: Mask
    score_map: Optional[ScoreMap]
    char_boxes: Sequence[CharBox]
    # NOTE: char_glyphs might not have the same shapes as char_boxes.
    char_glyphs: Sequence[CharGlyph]
    cv_resize_interpolation: int
    style: FontEngineRunConfigStyle
    font_size: int
    text: str
    is_hori: bool

    # Shifted text line is bound to a page.
    shifted: bool = False

    # For debugging.
    font_variant: Optional[FontVariant] = None

    @property
    def box(self):
        assert self.mask.box
        return self.mask.box

    @property
    def glyph_color(self):
        return self.style.glyph_color

    def to_shifted_text_line(self, offset_y: int = 0, offset_x: int = 0):
        self.shifted = True

        shifted_image = self.image.to_shifted_image(offset_y=offset_y, offset_x=offset_x)
        shifted_mask = self.mask.to_shifted_mask(offset_y=offset_y, offset_x=offset_x)

        shifted_score_map = None
        if self.score_map:
            shifted_score_map = self.score_map.to_shifted_score_map(
                offset_y=offset_y,
                offset_x=offset_x,
            )

        shifted_char_boxes = [
            char_box.to_shifted_char_box(
                offset_y=offset_y,
                offset_x=offset_x,
            ) for char_box in self.char_boxes
        ]

        return attrs.evolve(
            self,
            image=shifted_image,
            mask=shifted_mask,
            score_map=shifted_score_map,
            char_boxes=shifted_char_boxes,
        )

    def split(self):
        texts = self.text.split()
        if len(texts) == 1:
            # No need to split.
            return [self]
        assert len(texts) > 1

        # Seperated by space(s).
        text_lines: List[TextLine] = []

        begin = 0
        for text in texts:
            end = begin + len(text) - 1
            char_boxes = self.char_boxes[begin:end + 1]
            char_glyphs = self.char_glyphs[begin:end + 1]

            if self.is_hori:
                left = char_boxes[0].left
                right = char_boxes[-1].right
                up = min(char_box.up for char_box in char_boxes)
                down = max(char_box.down for char_box in char_boxes)
            else:
                up = char_boxes[0].up
                down = char_boxes[-1].down
                left = min(char_box.left for char_box in char_boxes)
                right = max(char_box.right for char_box in char_boxes)
            box = Box(up=up, down=down, left=left, right=right)

            image = box.extract_image(self.image)
            mask = box.extract_mask(self.mask)
            score_map = None
            if self.score_map:
                score_map = box.extract_score_map(self.score_map)

            text_lines.append(
                attrs.evolve(
                    self,
                    image=image,
                    mask=mask,
                    score_map=score_map,
                    char_boxes=char_boxes,
                    char_glyphs=char_glyphs,
                    text=text,
                )
            )
            begin = end + 1

        return text_lines

    def to_polygon(self):
        if self.is_hori:
            xs = [self.box.left]
            for char_box in self.char_boxes:
                if xs[-1] < char_box.left:
                    xs.append(char_box.left)
                if char_box.left < char_box.right:
                    xs.append(char_box.right)
            if xs[-1] < self.box.right:
                xs.append(self.box.right)

            points = PointList()

            for x in xs:
                points.append(Point.create(y=self.box.up, x=x))

            y_mid = (self.box.up + self.box.down) // 2
            if self.box.up < y_mid < self.box.down:
                points.append(Point.create(y=y_mid, x=xs[-1]))

            for x in reversed(xs):
                points.append(Point.create(y=self.box.down, x=x))

            if self.box.up < y_mid < self.box.down:
                points.append(Point.create(y=y_mid, x=xs[0]))

            return Polygon.create(points=points)

        else:
            ys = [self.box.up]
            for char_box in self.char_boxes:
                if ys[-1] < char_box.up:
                    ys.append(char_box.up)
                if char_box.up < char_box.down:
                    ys.append(char_box.down)
            if ys[-1] < self.box.down:
                ys.append(self.box.down)

            points = PointList()

            for y in ys:
                points.append(Point.create(y=y, x=self.box.right))

            x_mid = (self.box.left + self.box.right) // 2
            if self.box.left < x_mid < self.box.right:
                points.append(Point.create(y=ys[-1], x=x_mid))

            for y in reversed(ys):
                points.append(Point.create(y=y, x=self.box.left))

            if self.box.left < x_mid < self.box.right:
                points.append(Point.create(y=ys[0], x=x_mid))

            return Polygon.create(points=points)

    @classmethod
    def build_char_polygon(
        cls,
        up: float,
        down: float,
        left: float,
        right: float,
    ):
        return Polygon.from_xy_pairs([
            (left, up),
            (right, up),
            (right, down),
            (left, down),
        ])

    def to_char_polygons(self, page_height: int, page_width: int):
        assert len(self.char_boxes) == len(self.char_glyphs)

        if self.is_hori:
            polygons: List[Polygon] = []
            for char_box, char_glyph in zip(self.char_boxes, self.char_glyphs):
                ref_char_height = char_glyph.ref_char_height
                ref_char_width = char_glyph.ref_char_width
                box = char_box.box

                up = box.up
                down = box.down
                if box.height < ref_char_height:
                    inc = ref_char_height - box.height
                    half_inc = inc / 2
                    up = max(0, up - half_inc)
                    down = min(page_height - 1, down + half_inc)

                left = box.left
                right = box.right
                if box.width < ref_char_width:
                    inc = ref_char_width - box.width
                    half_inc = inc / 2
                    left = max(0, left - half_inc)
                    right = min(page_width - 1, right + half_inc)

                polygons.append(self.build_char_polygon(
                    up=up,
                    down=down,
                    left=left,
                    right=right,
                ))
            return polygons

        else:
            polygons: List[Polygon] = []
            for char_box, char_glyph in zip(self.char_boxes, self.char_glyphs):
                ref_char_height = char_glyph.ref_char_height
                ref_char_width = char_glyph.ref_char_width
                box = char_box.box

                left = box.left
                right = box.right
                if box.width < ref_char_height:
                    inc = ref_char_height - box.width
                    half_inc = inc / 2
                    left = max(0, left - half_inc)
                    right = min(page_width - 1, right + half_inc)

                up = box.up
                down = box.down
                if box.height < ref_char_width:
                    inc = ref_char_width - box.height
                    half_inc = inc / 2
                    up = max(self.box.up, up - half_inc)
                    down = min(page_height - 1, down + half_inc)

                polygons.append(self.build_char_polygon(
                    up=up,
                    down=down,
                    left=left,
                    right=right,
                ))
            return polygons

    def get_height_points(self, num_points: int, is_up: bool):
        if self.is_hori:
            step = max(1, self.box.width // num_points)
            xs = list(range(0, self.box.right + 1, step))
            if len(xs) >= num_points:
                xs = xs[:num_points - 1]
                xs.append(self.box.right)

            points = PointList()
            for x in xs:
                if is_up:
                    y = self.box.up
                else:
                    y = self.box.down
                points.append(Point.create(y=y, x=x))
            return points

        else:
            step = max(1, self.box.height // num_points)
            ys = list(range(self.box.up, self.box.down + 1, step))
            if len(ys) >= num_points:
                ys = ys[:num_points - 1]
                ys.append(self.box.down)

            points = PointList()
            for y in ys:
                if is_up:
                    x = self.box.right
                else:
                    x = self.box.left
                points.append(Point.create(y=y, x=x))
            return points

    def get_char_level_height_points(self, is_up: bool):
        if self.is_hori:
            points = PointList()
            for char_box in self.char_boxes:
                x = (char_box.left + char_box.right) / 2
                if is_up:
                    y = self.box.up
                else:
                    y = self.box.down
                points.append(Point.create(y=y, x=x))
            return points

        else:
            points = PointList()
            for char_box in self.char_boxes:
                y = (char_box.up + char_box.down) / 2
                if is_up:
                    x = self.box.right
                else:
                    x = self.box.left
                points.append(Point.create(y=y, x=x))
            return points
