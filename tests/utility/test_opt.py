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
from typing import Optional

import attrs

from vkit.utility import attrs_lazy_field


@attrs.define
class Foo:
    a: int
    _b: Optional[int] = attrs_lazy_field()

    @property
    def b(self):
        if self._b is None:
            self._b = self.a + 1
        return self._b


def test_attrs_lazy_field():
    foo0 = Foo(42)
    assert foo0.b == 43
    foo1 = attrs.evolve(foo0)
    assert foo1._b is None  # type: ignore
    assert foo1.b == 43
    foo2 = attrs.evolve(foo0, a=1)
    assert foo2._b is None  # type: ignore
    assert foo2.b == 2
