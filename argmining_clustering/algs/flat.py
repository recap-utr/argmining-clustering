import typing as t

import numpy as np


def run(
    sim_matrix: np.ndarray, mc_index: t.Optional[int] = None
) -> t.Tuple[int, t.List[t.Tuple[int, int]]]:
    if not mc_index:
        mc_index = t.cast(int, sim_matrix.sum(axis=1).argmax())

    relations = [
        (premise_index, mc_index)
        for premise_index in range(sim_matrix.shape[1])
        if premise_index != mc_index
    ]

    return (mc_index, relations)
