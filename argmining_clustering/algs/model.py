import typing as t
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Relation:
    premise: int
    claim: int


Relations = t.List[Relation]


@dataclass(frozen=True, eq=True)
class Result:
    mc: int
    relations: Relations
