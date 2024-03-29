import typing as t

import numpy as np
import numpy.typing as npt
from argmining_clustering.algs.model import Relation, Result


def run(sim_matrix: npt.NDArray[np.float_], mc_index: t.Optional[int] = None) -> Result:
    if mc_index is None:
        mc_index = t.cast(int, sim_matrix.sum(axis=1).argmax())

    relations = [
        Relation(premise_index, mc_index)
        for premise_index in range(sim_matrix.shape[1])
        if premise_index != mc_index
    ]

    return Result(mc_index, relations)
