# TODO
# https://github.com/microsoft/pyright/issues/3954
# from typing import Callable, TypeVar
# from typing_extensions import Concatenate, ParamSpec

# _S = TypeVar('_S')
# _P = ParamSpec('_P')
# _R = TypeVar('_R')

# def proxy_method(method: Callable[[_S], Callable[_P, _R]]) -> Callable[Concatenate[_S, _P], _R]:

#     def new_method(_self: _S, *args: _P.args, **kwargs: _P.kwargs):
#         other_method = method(_self)
#         return other_method(*args, **kwargs)

#     return new_method

# class Foo:

#     def do_something(self, x: int):
#         return 42 + x

# class Bar:

#     def __init__(self):
#         self.foo = Foo()

#     @proxy_method
#     def do_something(self):
#         return self.foo.do_something

#     def do_other(self, x: int):
#         return x + 1

# def debug():
#     Bar.do_something
#     Bar.do_other
#     bar = Bar()
#     print(bar.do_other(1))
#     print(bar.do_something(1))
