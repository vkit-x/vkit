from typing import Tuple
from enum import Enum, unique


class Shapable:

    @property
    def height(self) -> int:
        raise NotImplementedError()

    @property
    def width(self) -> int:
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width


@unique
class FillByElementsMode(Enum):
    UNION = 'union'
    DISTINCT = 'distinct'
    INTERSECT = 'intersect'
