from typing import (
    Any,
    Callable,
    Dict,
    List,
    NewType,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from collections.abc import Collection

DInt = Union[int, Dict[str, int]]
DShape = Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]
Shape = Union[DInt, DShape]
