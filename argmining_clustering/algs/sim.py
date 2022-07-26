import itertools
import typing as t

import numpy as np
import numpy.typing as npt
from argmining_clustering.algs.model import Relation, Result


def run(sim_matrix: npt.NDArray[np.float_], mc_index: t.Optional[int] = None) -> Result:
    if not mc_index:
        mc_index = t.cast(int, sim_matrix.sum(axis=1).argmax())

    relations = []
    assigned_indices = [mc_index]
    remaining_indices = [i for i in range(sim_matrix.shape[0]) if i != mc_index]

    while len(remaining_indices) > 0:
        similarities: dict[tuple[int, int], float] = {}

        for premise, claim in itertools.product(remaining_indices, assigned_indices):
            similarities[(premise, claim)] = sim_matrix[premise, claim]

        (premise, claim), sim = max(similarities.items(), key=lambda x: x[1])

        relations.append(Relation(premise, claim))
        remaining_indices.remove(premise)
        assigned_indices.append(premise)

    return Result(mc_index, relations)
