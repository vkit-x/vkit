from typing import Tuple


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
