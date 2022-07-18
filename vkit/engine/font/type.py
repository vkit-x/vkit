from typing import Sequence, List, Optional, Mapping, Dict, DefaultDict, Iterable, Set, Tuple
from enum import Enum, unique
from collections import defaultdict

import attrs
import cattrs
import iolite as io

from vkit.utility import PathType, dyn_structure
from vkit.element import (
    Image,
    Box,
    CharBox,
    Mask,
    ScoreMap,
    Point,
    PointList,
    Polygon,
)


@unique
class FontKind(Enum):
    # ttc file.
    TTC = 'ttc'
    # Grouped ttf file(s).
    VTTC = 'vttc'
    # Grouped otf file(s).
    VOTC = 'votc'


@attrs.define
class FontVariant:
    font_file: PathType
    ascent_plus_pad_up_min_to_font_size_ratio: float
    height_min_to_font_size_ratio: float
    width_min_to_font_size_ratio: float
    is_ttc: bool = False
    ttc_font_index: Optional[int] = None


@attrs.define
class FontMeta:
    name: str
    kind: FontKind
    chars: Sequence[str]
    font_files: Sequence[str]
    ascent_plus_pad_up_min_to_font_size_ratio: float
    height_min_to_font_size_ratio: float
    width_min_to_font_size_ratio: float
    # NOTE: ttc_font_index_max is inclusive.
    ttc_font_index_max: Optional[int] = None

    def __repr__(self):
        return (
            'FontMeta('
            f'name="{self.name}", '
            f'kind={self.kind}, '
            f'num_chars={len(self.chars)}), '
            f'font_files={self.font_files}, '
            f'ttc_font_index_max={self.ttc_font_index_max})'
        )

    @staticmethod
    def from_file(
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

        io.write_json(path, cattrs.unstructure(font), indent=2, ensure_ascii=False)

    @property
    def num_font_variants(self):
        if self.kind in (FontKind.VOTC, FontKind.VTTC):
            return len(self.font_files)

        elif self.kind == FontKind.TTC:
            assert self.ttc_font_index_max is not None
            return self.ttc_font_index_max + 1

        else:
            raise NotImplementedError()

    def get_font_variant(self, variant_idx: int):
        if self.kind in (FontKind.VOTC, FontKind.VTTC):
            assert variant_idx < len(self.font_files)
            return FontVariant(
                font_file=io.file(self.font_files[variant_idx]),
                ascent_plus_pad_up_min_to_font_size_ratio=(
                    self.ascent_plus_pad_up_min_to_font_size_ratio
                ),
                height_min_to_font_size_ratio=self.height_min_to_font_size_ratio,
                width_min_to_font_size_ratio=self.width_min_to_font_size_ratio,
            )

        elif self.kind == FontKind.TTC:
            assert self.ttc_font_index_max is not None
            assert variant_idx <= self.ttc_font_index_max
            return FontVariant(
                font_file=io.file(self.font_files[0]),
                ascent_plus_pad_up_min_to_font_size_ratio=(
                    self.ascent_plus_pad_up_min_to_font_size_ratio
                ),
                height_min_to_font_size_ratio=self.height_min_to_font_size_ratio,
                width_min_to_font_size_ratio=self.width_min_to_font_size_ratio,
                is_ttc=True,
                ttc_font_index=variant_idx,
            )

        else:
            raise NotImplementedError()


@unique
class FontCollectionFolderTree(Enum):
    FONT = 'font'
    FONT_META = 'font_meta'


@attrs.define
class FontCollection:
    font_metas: Sequence[FontMeta]

    _name_to_font_meta: Optional[Mapping[str, FontMeta]] = None
    _char_to_font_meta_names: Optional[Mapping[str, Set[str]]] = None

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

    @staticmethod
    def from_folder(folder: PathType):
        in_fd = io.folder(folder, expandvars=True, exists=True)
        font_fd = io.folder(in_fd / FontCollectionFolderTree.FONT.value, exists=True)
        font_meta_fd = io.folder(in_fd / FontCollectionFolderTree.FONT_META.value, exists=True)

        font_metas: List[FontMeta] = []
        for font_meta_json in font_meta_fd.glob('*.json'):
            font_metas.append(FontMeta.from_file(font_meta_json, font_fd))

        return FontCollection(font_metas=font_metas)


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


@attrs.define
class TextLine:
    image: Image
    mask: Mask
    score_map: Optional[ScoreMap]
    glyph_color: Tuple[int, int, int]
    char_boxes: Sequence[CharBox]
    font_size: int
    ref_char_height: int
    ref_char_width: int
    text: str
    is_hori: bool

    # For debugging.
    font_variant: Optional[FontVariant] = None

    @property
    def box(self):
        assert self.mask.box
        return self.mask.box

    def to_shifted_text_line(self, y_offset: int = 0, x_offset: int = 0):
        shifted_image = self.image.to_shifted_image(y_offset=y_offset, x_offset=x_offset)
        shifted_mask = self.mask.to_shifted_mask(y_offset=y_offset, x_offset=x_offset)

        shifted_score_map = None
        if self.score_map:
            shifted_score_map = self.score_map.to_shifted_score_map(
                y_offset=y_offset,
                x_offset=x_offset,
            )

        shifted_char_boxes = [
            char_box.to_shifted_char_box(
                y_offset=y_offset,
                x_offset=x_offset,
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
                points.append(Point(y=self.box.up, x=x))

            y_mid = (self.box.up + self.box.down) // 2
            if self.box.up < y_mid < self.box.down:
                points.append(Point(y=y_mid, x=xs[-1]))

            for x in reversed(xs):
                points.append(Point(y=self.box.down, x=x))

            if self.box.up < y_mid < self.box.down:
                points.append(Point(y=y_mid, x=xs[0]))

            return Polygon(points=points)

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
                points.append(Point(y=y, x=self.box.right))

            x_mid = (self.box.left + self.box.right) // 2
            if self.box.left < x_mid < self.box.right:
                points.append(Point(y=ys[-1], x=x_mid))

            for y in reversed(ys):
                points.append(Point(y=y, x=self.box.left))

            if self.box.left < x_mid < self.box.right:
                points.append(Point(y=ys[0], x=x_mid))

            return Polygon(points=points)

    def to_char_polygons(
        self,
        page_height: int,
        page_width: int,
        ref_char_height_ratio: float = 1.0,
        ref_char_width_ratio: float = 1.0,
    ):
        ref_char_height = round(self.ref_char_height * ref_char_height_ratio)
        ref_char_width = round(self.ref_char_width * ref_char_width_ratio)

        if self.is_hori:
            polygons: List[Polygon] = []
            for char_box in self.char_boxes:
                box = char_box.box

                up = box.up
                down = box.down
                if box.height < ref_char_height:
                    inc = ref_char_height - box.height
                    inc_up = inc // 2
                    inc_down = inc - inc_up
                    up = max(0, up - inc_up)
                    down = min(page_height - 1, down + inc_down)

                left = box.left
                right = box.right
                if box.width < ref_char_width:
                    inc = ref_char_width - box.width
                    inc_left = inc // 2
                    inc_right = inc - inc_left
                    left = max(0, left - inc_left)
                    right = min(page_width - 1, right + inc_right)

                box = Box(up=up, down=down, left=left, right=right)
                polygons.append(box.to_polygon())
            return polygons

        else:
            polygons: List[Polygon] = []
            for char_box in self.char_boxes:
                box = char_box.box

                left = box.left
                right = box.right
                if box.width < ref_char_height:
                    inc = ref_char_height - box.width
                    inc_left = inc // 2
                    inc_right = inc - inc_left
                    left = max(0, left - inc_left)
                    right = min(page_width - 1, right + inc_right)

                up = box.up
                down = box.down
                if box.height < ref_char_width:
                    inc = ref_char_width - box.height
                    inc_up = inc // 2
                    inc_down = inc - inc_up
                    up = max(self.box.up, up - inc_up)
                    down = min(page_height - 1, down + inc_down)

                box = Box(up=up, down=down, left=left, right=right)
                polygons.append(box.to_polygon())
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
                points.append(Point(y=y, x=x))
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
                points.append(Point(y=y, x=x))
            return points

    def get_char_level_height_points(self, is_up: bool):
        if self.is_hori:
            points = PointList()
            for char_box in self.char_boxes:
                x = (char_box.left + char_box.right) // 2
                if is_up:
                    y = self.box.up
                else:
                    y = self.box.down
                points.append(Point(y=y, x=x))
            return points

        else:
            points = PointList()
            for char_box in self.char_boxes:
                y = (char_box.up + char_box.down) // 2
                if is_up:
                    x = self.box.right
                else:
                    x = self.box.left
                points.append(Point(y=y, x=x))
            return points
