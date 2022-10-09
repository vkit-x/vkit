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
from typing import Mapping, Sequence, Optional, DefaultDict, List
from collections import defaultdict
import hashlib

import attrs
import cattrs
import iolite as io

from vkit.utility import attrs_lazy_field, dyn_structure, PathType


@attrs.define(frozen=True)
class Lexicon:
    char: str
    aliases: Sequence[str] = attrs.field(factory=tuple)
    tags: Sequence[str] = attrs.field(factory=tuple)
    meta: Optional[Mapping[str, str]] = None

    def __attrs_post_init__(self):
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "tags", tuple(self.tags))

    @property
    def char_and_aliases(self):
        return [self.char, *self.aliases]

    @property
    def unicode_id(self):
        return hex(ord(self.char)).upper()[2:]


KEY_NO_TAG = '__no_tag'


@attrs.define
class LexiconCollection:
    lexicons: Sequence[Lexicon]

    _char_to_lexicon: Optional[Mapping[str, Lexicon]] = attrs_lazy_field()
    _tag_to_lexicons: Optional[Mapping[str, Sequence[Lexicon]]] = attrs_lazy_field()
    _tags: Optional[Sequence[str]] = attrs_lazy_field()

    def lazy_post_init(self):
        initialized = (self._char_to_lexicon is not None)
        if initialized:
            return

        self._char_to_lexicon = {}
        for lexicon in self.lexicons:
            for char in lexicon.char_and_aliases:
                assert char not in self._char_to_lexicon
                self._char_to_lexicon[char] = lexicon

        tag_to_lexicons: DefaultDict[str, List[Lexicon]] = defaultdict(list)
        for lexicon in self.lexicons:
            if lexicon.tags:
                for tag in lexicon.tags:
                    tag_to_lexicons[tag].append(lexicon)
            else:
                tag_to_lexicons[KEY_NO_TAG].append(lexicon)
        self._tag_to_lexicons = dict(tag_to_lexicons)
        self._tags = sorted(self._tag_to_lexicons)

    @property
    def char_to_lexicon(self):
        self.lazy_post_init()
        assert self._char_to_lexicon is not None
        return self._char_to_lexicon

    @property
    def tag_to_lexicons(self):
        self.lazy_post_init()
        assert self._tag_to_lexicons is not None
        return self._tag_to_lexicons

    @property
    def tags(self):
        self.lazy_post_init()
        assert self._tags is not None
        return self._tags

    def has_char(self, char: str):
        return char in self.char_to_lexicon

    def get_lexicon(self, char: str):
        return self.char_to_lexicon[char]

    @classmethod
    def from_file(cls, path: PathType):
        lexicons = dyn_structure(path, Sequence[Lexicon], force_path_type=True)
        return cls(lexicons=lexicons)

    def to_file(self, path: PathType):
        io.write_json(
            path,
            cattrs.unstructure(self.lexicons),
            indent=2,
            ensure_ascii=False,
        )

    def get_hash(self):
        sha256_algo = hashlib.sha256()
        for lexicon in self.lexicons:
            sha256_algo.update(lexicon.char.encode())
            for alias in lexicon.aliases:
                sha256_algo.update(alias.encode())
        return sha256_algo.hexdigest()
