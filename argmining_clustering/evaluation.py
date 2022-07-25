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


def error(method: str, metric: str, graph1: ag.Graph, graph2: ag.Graph) -> float:
    print(f"{method}: Error for {metric}({graph1.name}, {graph2.name})")
    return 0.0


def avg(method: str, graphs: t.List[t.Tuple[ag.Graph, ag.Graph]]) -> dict[str, float]:
    global_dist = {}

    for func in FUNCTIONS:
        local_dist = []

        for (graph1, graph2) in graphs:
            try:
                local_dist.append(func(graph1, graph2))
            except ValueError:
                local_dist.append(error(method, func.__name__, graph1, graph2))

        global_dist[func.__name__] = mean(local_dist)

    return global_dist


def edit_graphkit_learn(graph1: ag.Graph, graph2: ag.Graph) -> float:
    """https://github.com/jajupmochi/graphkit-learn/blob/master/gklearn/examples/ged/compute_graph_edit_distance.py"""

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

    return 1 - ged_env.get_upper_bound(g1, g2) / _normalization(graph1, graph2)


def edit_graphmatch(graph1: ag.Graph, graph2: ag.Graph) -> float:
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
    return 1 - np.min(operations[np.nonzero(operations)]) / _normalization(
        graph1, graph2
    )


def edit_nx(graph1: ag.Graph, graph2: ag.Graph) -> float:
    nx1 = graph1.to_nx(**NX_OPT)
    nx2 = graph2.to_nx(**NX_OPT)

    if dist := nx.graph_edit_distance(
        nx1,
        nx2,
        node_match=lambda x, y: x["label"] == y["label"],
    ):
        return 1 - dist / _normalization(graph1, graph2)

    return 0.0


def jaccard_nodes(graph1: ag.Graph, graph2: ag.Graph) -> float:
    atoms = graph1.atom_nodes.keys()

    incoming_nodes_1 = [
        {node.id for node in graph1.incoming_nodes(node_id)} for node_id in atoms
    ]
    incoming_nodes_2 = [
        {node.id for node in graph2.incoming_nodes(node_id)} for node_id in atoms
    ]

    mlb = MultiLabelBinarizer(classes=list(atoms))

    return jaccard_score(
        mlb.fit_transform(incoming_nodes_1),
        mlb.fit_transform(incoming_nodes_2),
        average="macro",
        zero_division=1,  # type: ignore
    )


def jaccard_edges(graph1: ag.Graph, graph2: ag.Graph) -> float:
    edges1 = {f"{e.source.id},{e.target.id}" for e in graph1.edges.values()}
    edges2 = {f"{e.source.id},{e.target.id}" for e in graph2.edges.values()}

    # https://www.nltk.org/_modules/nltk/metrics/distance.html#jaccard_distance
    return 1 - (
        (len(edges1.union(edges2)) - len(edges1.intersection(edges2)))
        / len(edges1.union(edges2))
    )


def _normalization(graph1: ag.Graph, graph2: ag.Graph) -> float:
    return len(graph1.nodes) + len(graph2.nodes) + len(graph1.edges) + len(graph2.edges)


FUNCTIONS = [edit_graphkit_learn, jaccard_nodes, jaccard_edges, edit_graphmatch]
