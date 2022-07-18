import pytest
import attrs
import cattrs


@attrs.define
class Foo:
    a: int = 42


def test_invalid_field():
    cattrs.structure({'b': 43}, Foo)

    with pytest.raises(Exception):
        cvt = cattrs.GenConverter(forbid_extra_keys=True)
        cvt.structure({'b': 43}, Foo)
