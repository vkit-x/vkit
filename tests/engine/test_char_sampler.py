import tempfile
import string

from numpy.random import default_rng
import pytest

from vkit.element import LexiconCollection, Lexicon
from vkit.engine.char_sampler import *


def test_corpus_sampler():
    lexicon_collection = LexiconCollection(
        lexicons=[
            Lexicon(char='a'),
            Lexicon(char='b'),
            Lexicon(char='c'),
            Lexicon(char='e'),
            Lexicon(char='f'),
        ]
    )
    text = '''
abcdef
cba
nba

ddddd
abbb cd dd ef
'''
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(text.encode())
        temp_file.flush()
        corpus_sampler = corpus_char_sampler_factory.create(
            {'txt_files': [temp_file.name]},
            {'lexicon_collection': lexicon_collection},
        )

        rng = default_rng(0)
        chars = corpus_sampler.run(CharSamplerEngineRunConfig(5), rng=rng)
        assert ''.join(chars) == 'abbbc'
        chars = corpus_sampler.run(CharSamplerEngineRunConfig(10), rng=rng)
        assert ''.join(chars) == 'ba cba cba'


def test_datetime_sampler():
    lexicon_collection = LexiconCollection(
        lexicons=[Lexicon(char=char) for char in string.printable if not char.isspace()]
    )
    datetime_sampler = datetime_char_sampler_factory.create(
        {
            'datetime_formats': ['%Y-%m-%d %H:%M:%S %Z%z'],
            'timezones': ['Europe/Athens'],
        },
        {'lexicon_collection': lexicon_collection},
    )

    rng = default_rng(0)
    chars = datetime_sampler.run(CharSamplerEngineRunConfig(40), rng=rng)
    assert ''.join(chars) == '2042.03.08 01:20:22 EET+0200 2007-11-261'


@pytest.mark.local
def test_faker_sampler():
    lexicon_collection = LexiconCollection.from_file(
        '$VKIT_PRIVATE_DATA/vkit_lexicon/lexicon_collection_combined/chinese.json'
    )
    faker_sampler = faker_char_sampler_factory.create(
        resource={'lexicon_collection': lexicon_collection}
    )
    rng = default_rng(0)
    chars = faker_sampler.run(CharSamplerEngineRunConfig(20), rng=rng)
    assert ''.join(chars) == '011 80198344 +1 246青'


@pytest.mark.local
def test_lexicon_sampler():
    lexicon_collection = LexiconCollection.from_file(
        '$VKIT_PRIVATE_DATA/vkit_lexicon/lexicon_collection_combined/chinese.json'
    )
    lexicon_sampler = lexicon_char_sampler_factory.create(
        {'space_prob': 0.1},
        {'lexicon_collection': lexicon_collection},
    )
    rng = default_rng(0)
    chars = lexicon_sampler.run(CharSamplerEngineRunConfig(20), rng=rng)
    assert ''.join(chars) == '硜柳展 潽鷫 欝唁巫鵝冐䓨鲱戳瘋圇朐 録'


def test_create_char_sampler_factor_from_file():
    text = '''
[
    {
        "type": "datetime",
        "weight": 1,
        "config": {
            "datetime_formats": [
                "%Y-%m-%d %H:%M:%S %Z%z"
            ],
            "timezones": [
                "Europe/Athens"
            ]
        }
    }
]
'''
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(text.encode())
        temp_file.flush()

        lexicon_collection = LexiconCollection(
            lexicons=[Lexicon(char=char) for char in string.printable if not char.isspace()]
        )
        char_sampler_aggregator = char_sampler_factory.create(
            temp_file.name,
            {'lexicon_collection': lexicon_collection},
        )
        rng = default_rng(0)
        chars = char_sampler_aggregator.run({'num_chars': 40}, rng=rng)
        assert ''.join(chars) == '2022-02-23 01-23-28 EET+0200 1994-05-260'
