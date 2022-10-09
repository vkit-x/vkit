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
import tempfile
import string
from os.path import getsize

from numpy.random import default_rng
import iolite as io
import pytest

from vkit.element import LexiconCollection, Lexicon
from vkit.engine.char_sampler import *


@pytest.mark.local
def test_corpus_sampler_sample_text_line_from_file():
    rng = default_rng(42)
    txt_file = io.file(
        '$VKIT_PRIVATE_DATA/char_sampler/corp-address-debug.txt',
        expandvars=True,
        exists=True,
    )
    size = getsize(txt_file)
    text = CharSamplerCorpusEngine.sample_text_line_from_file(
        txt_file,
        size,
        rng,
    )
    assert text == '北京市昌平区回龙观镇回龙观西大街9号院4号楼14层1703'


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
        corpus_sampler = char_sampler_corpus_engine_executor_factory.create(
            {'txt_files': [temp_file.name]},
            {'lexicon_collection': lexicon_collection},
        )

        rng = default_rng(0)
        chars = corpus_sampler.run(CharSamplerEngineRunConfig(5), rng=rng)
        assert ''.join(chars) == 'cba a'
        chars = corpus_sampler.run(CharSamplerEngineRunConfig(10), rng=rng)
        assert ''.join(chars) == 'abbb c efa'


def test_datetime_sampler():
    lexicon_collection = LexiconCollection(
        lexicons=[Lexicon(char=char) for char in string.printable if not char.isspace()]
    )
    datetime_sampler = char_sampler_datetime_engine_executor_factory.create(
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
    faker_sampler = char_sampler_faker_engine_executor_factory.create(
        init_resource={'lexicon_collection': lexicon_collection}
    )
    rng = default_rng(0)
    chars = faker_sampler.run(CharSamplerEngineRunConfig(20), rng=rng)
    assert ''.join(chars) == '011 80198344 +1 246青'


@pytest.mark.local
def test_lexicon_sampler():
    lexicon_collection = LexiconCollection.from_file(
        '$VKIT_PRIVATE_DATA/vkit_lexicon/lexicon_collection_combined/chinese.json'
    )
    lexicon_sampler = char_sampler_lexicon_engine_executor_factory.create(
        {'prob_space': 0.1},
        {'lexicon_collection': lexicon_collection},
    )
    rng = default_rng(0)
    chars = lexicon_sampler.run(CharSamplerEngineRunConfig(20), rng=rng)
    assert ''.join(chars) == '硜柳展 潽鷫 欝唁巫鵝冐䓨鲱戳瘋圇朐 録'


def test_create_char_sampler_factor_from_file():
    temp_txt_file = tempfile.NamedTemporaryFile()
    text = '''
a
b
c
'''
    temp_txt_file.write(text.encode())
    temp_txt_file.flush()

    temp_config_file = tempfile.NamedTemporaryFile()
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
    },
    {
        "type": "corpus",
        "weight": 1,
        "config": {
            "txt_files": ["PLACEHOLDER"]
        }
    }
]
'''
    text = text.replace('PLACEHOLDER', temp_txt_file.name)
    temp_config_file.write(text.encode())
    temp_config_file.flush()

    lexicon_collection = LexiconCollection(
        lexicons=[Lexicon(char=char) for char in string.printable if not char.isspace()]
    )
    char_sampler_engine_executor_aggregator = \
        char_sampler_engine_executor_aggregator_factory.create_with_repeated_init_resource(
            temp_config_file.name,
            {'lexicon_collection': lexicon_collection},
        )

    rng = default_rng(0)
    chars = char_sampler_engine_executor_aggregator.run({'num_chars': 40}, rng=rng)
    assert ''.join(chars) == 'b a c c b c a c c a b b a b c b c a b ba'

    rng = default_rng(0)
    chars = char_sampler_engine_executor_aggregator.run(
        {
            'num_chars': 40,
            'enable_aggregator_mode': True
        },
        rng=rng,
    )
    assert ''.join(chars) == 'b ab2047-03-02 10 06 29 EET+0200 a2042-1'

    temp_txt_file.close()
    temp_config_file.close()
