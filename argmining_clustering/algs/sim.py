import typing as t

import numpy as np
from argmining_clustering.algs.model import Result


def run(sim_matrix: np.ndarray, mc_index: t.Optional[int] = None) -> Result:
    if not mc_index:
        mc_index = t.cast(int, sim_matrix.sum(axis=1).argmax())

    relations = []

    # TODO

    return Result(mc_index, relations)
