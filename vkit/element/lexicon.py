from typing import Dict, Sequence, Optional, DefaultDict, List
from collections import defaultdict
import hashlib

import attrs
import cattrs
import iolite as io

from vkit.utility import PathType, read_json_file


@attrs.define(frozen=True)
class Lexicon:
    char: str
    aliases: Sequence[str] = attrs.field(factory=tuple)
    tags: Sequence[str] = attrs.field(factory=tuple)
    meta: Optional[Dict[str, str]] = None

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

    char_to_lexicon: Dict[str, Lexicon] = attrs.field(init=False)
    tag_to_lexicons: Dict[str, Sequence[Lexicon]] = attrs.field(init=False)
    tags: Sequence[str] = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.char_to_lexicon = {}
        for lexicon in self.lexicons:
            for char in lexicon.char_and_aliases:
                assert char not in self.char_to_lexicon
                self.char_to_lexicon[char] = lexicon

        tag_to_lexicons: DefaultDict[str, List[Lexicon]] = defaultdict(list)
        for lexicon in self.lexicons:
            if lexicon.tags:
                for tag in lexicon.tags:
                    tag_to_lexicons[tag].append(lexicon)
            else:
                tag_to_lexicons[KEY_NO_TAG].append(lexicon)
        self.tag_to_lexicons = dict(tag_to_lexicons)
        self.tags = sorted(self.tag_to_lexicons)

    def has_char(self, char: str):
        return char in self.char_to_lexicon

    def get_lexicon(self, char: str):
        return self.char_to_lexicon[char]

    @staticmethod
    def from_file(path: PathType):
        lexicons = cattrs.structure(read_json_file(path), Sequence[Lexicon])
        return LexiconCollection(lexicons=lexicons)

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
