import typing as t
from pathlib import Path
from statistics import mean

import arguebuf as ag
import graphmatch as gm
import networkx as nx
import numpy as np
import numpy.typing as npt
from gklearn.ged.env import GEDEnv
from gklearn.ged.env import Options as GEDOptions
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

NX_OPT = {"atom_attrs": {"label": lambda x: x.id}}


def error(metric: str, graph1: ag.Graph, graph2: ag.Graph) -> float:
    print(
        f"Cannot compute distance '{metric}' for graphs '{graph1.name}' ({len(graph1.nodes)} nodes) and '{graph2.name}' ({len(graph2.nodes)} nodes)."
    )
    return len(graph1.nodes) + len(graph2.nodes)


def avg(graphs: t.List[t.Tuple[ag.Graph, ag.Graph]]) -> dict[str, float]:
    return {
        func.__name__: mean(func(graph1, graph2) for graph1, graph2 in graphs)
        for func in FUNCTIONS
    }


def edit_ged(graph1: ag.Graph, graph2: ag.Graph) -> float:
    """https://github.com/jajupmochi/graphkit-learn/blob/master/gklearn/examples/ged/compute_graph_edit_distance.py"""

    try:
        ged_env = GEDEnv()
        ged_env.set_edit_cost("CONSTANT", edit_cost_constants=[])
        g1 = ged_env.add_nx_graph(graph1.to_nx(), "")
        g2 = ged_env.add_nx_graph(graph2.to_nx(), "")

        ged_env.init(init_type=GEDOptions.InitType.LAZY_WITHOUT_SHUFFLED_COPIES)
        options = {
            # "initialization_method": "RANDOM",
            # "threads": 1,
        }
        ged_env.set_method(GEDOptions.GEDMethod.BIPARTITE, options)  # type: ignore
        ged_env.init_method()
        ged_env.run_method(g1, g2)

        ged_env.get_forward_map(g1, g2)
        ged_env.get_backward_map(g1, g2)

        return ged_env.get_upper_bound(g1, g2)

    except ValueError:
        return error("edit_ged", graph1, graph2)


def edit_gm(graph1: ag.Graph, graph2: ag.Graph) -> float:
    nx1 = graph1.to_nx(**NX_OPT)
    nx2 = graph2.to_nx(**NX_OPT)

    ged = gm.GraphEditDistance(1, 1, 1, 1)
    # ged.set_attr_graph_used("label", "")
    operations: npt.NDArray[np.float_] = ged.compare([nx1, nx2], None)

    # masked_operations: npt.NDArray[np.float_] = np.ma.masked_equal(
    #     operations, 0.0, copy=False
    # )
    # return masked_operations.min()

    # Return the minimum number that is NOT 0

    try:
        return np.min(operations[np.nonzero(operations)])

    except ValueError:
        return error("edit_gm", graph1, graph2)


def edit_nx(graph1: ag.Graph, graph2: ag.Graph) -> float:
    nx1 = graph1.to_nx(**NX_OPT)
    nx2 = graph2.to_nx(**NX_OPT)

    return (
        nx.graph_edit_distance(
            nx1,
            nx2,
            node_match=lambda x, y: x["label"] == y["label"],
        )
        or 0
    )


def jaccard(graph1: ag.Graph, graph2: ag.Graph) -> float:
    atoms = graph1.atom_nodes.keys()

    incoming_nodes_1 = [
        {node.id for node in graph1.incoming_nodes(node_id)} for node_id in atoms
    ]
    incoming_nodes_2 = [
        {node.id for node in graph2.incoming_nodes(node_id)} for node_id in atoms
    ]

    mlb = MultiLabelBinarizer(classes=list(atoms))

    return 1 - jaccard_score(
        mlb.fit_transform(incoming_nodes_1),
        mlb.fit_transform(incoming_nodes_2),
        average="macro",
        zero_division=1,  # type: ignore
    )


FUNCTIONS = [edit_ged, jaccard, edit_gm]
