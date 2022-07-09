import typing as t

import numpy as np
from argmining_clustering.algs.model import Result


def run(sim_matrix: np.ndarray, mc_index: t.Optional[int] = None) -> Result:
    if not mc_index:
        mc_index = t.cast(int, sim_matrix.sum(axis=1).argmax())

    relations = []

    assigned_indices = [mc_index]
    remaining_indices = [i for i in range(sim_matrix.shape[0]) if i != mc_index]

    while len(remaining_indices) > 0:
        pass

    return Result(mc_index, relations)
