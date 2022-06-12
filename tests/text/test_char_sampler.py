import tempfile
import string

from numpy.random import RandomState
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
            {'txt_file': temp_file.name},
            {'lexicon_collection': lexicon_collection},
        )

        rnd = RandomState(0)
        chars = corpus_sampler.run(CharSamplerEngineRunConfig(5), rnd=rnd)
        assert ''.join(chars) == 'abbbc'
        chars = corpus_sampler.run(CharSamplerEngineRunConfig(10), rnd=rnd)
        assert ''.join(chars) == 'abcef cbab'


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

    rnd = RandomState(0)
    chars = datetime_sampler.run(CharSamplerEngineRunConfig(40), rnd=rnd)
    assert ''.join(chars) == '1998 08 16 12:46:36 EEST+0300 2038.11.01'


@pytest.mark.local
def test_faker_sampler():
    lexicon_collection = LexiconCollection.from_file(
        '$VKIT_PRIVATE_DATA/vkit_lexicon/lexicon_collection_combined/chinese.json'
    )
    faker_sampler = faker_char_sampler_factory.create(
        resource={'lexicon_collection': lexicon_collection}
    )
    rnd = RandomState(0)
    chars = faker_sampler.run(CharSamplerEngineRunConfig(20), rnd=rnd)
    assert ''.join(chars) == '舒詩涵 843-939-3404 Sar'


@pytest.mark.local
def test_lexicon_sampler():
    lexicon_collection = LexiconCollection.from_file(
        '$VKIT_PRIVATE_DATA/vkit_lexicon/lexicon_collection_combined/chinese.json'
    )
    lexicon_sampler = lexicon_char_sampler_factory.create(
        {'space_prob': 0.1},
        {'lexicon_collection': lexicon_collection},
    )
    rnd = RandomState(0)
    chars = lexicon_sampler.run(CharSamplerEngineRunConfig(20), rnd=rnd)
    assert ''.join(chars) == '榃镕糁钡ⓨ珑鋶逍 马獅尟掽菖潵宋窎 罶荟'


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
        rnd = RandomState(0)
        chars = char_sampler_aggregator.run({'num_chars': 40}, rnd=rnd)
        assert ''.join(chars) == '2021-04-08 02.54.45 EEST+0300 2040:09:11'
