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
from rich import print
from sklearn.metrics import jaccard_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer


def error(method: str, metric: str, graph1: ag.Graph, graph2: ag.Graph) -> float:
    print(f"[b]{method}:[/b] Error for {metric}({graph1.name}, {graph2.name})")
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
    ged = gm.GraphEditDistance(1, 1, 1, 1)
    # ged.set_attr_graph_used("label", "")
    operations: npt.NDArray[np.float_] = ged.compare(
        [graph1.to_nx(), graph2.to_nx()], None
    )

    # masked_operations: npt.NDArray[np.float_] = np.ma.masked_equal(
    #     operations, 0.0, copy=False
    # )
    # return masked_operations.min()

    # Return the minimum number that is NOT 0
    return 1 - np.min(operations[np.nonzero(operations)]) / _normalization(
        graph1, graph2
    )


def edit_networkx(graph1: ag.Graph, graph2: ag.Graph) -> float:
    if dist := nx.graph_edit_distance(
        graph1.to_nx(atom_attrs={"label": lambda x: x.plain_text}),
        graph2.to_nx(atom_attrs={"label": lambda x: x.plain_text}),
        timeout=10,
        node_match=lambda x, y: x["label"] == y["label"],
        # node_subst_cost=lambda x, y: 0.0,
        # node_del_cost=lambda x: 0.0,
        # node_ins_cost=lambda x: 0.0,
        # edge_subst_cost=lambda x, y: 0.0,
        # edge_del_cost=lambda x: 1.0,
        # edge_ins_cost=lambda x: 1.0,
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


def visual_hierarchy(graph1: ag.Graph, graph2: ag.Graph) -> float:
    true_levels = _build_hierarchy(graph1)
    pred_levels = _build_hierarchy(graph2)

    # Pad the lists with zeros to have the same length
    max_len = max(len(true_levels), len(pred_levels))
    true_levels += [0] * (max_len - len(true_levels))
    pred_levels += [0] * (max_len - len(pred_levels))

    # Normalize the error and return it as a similarity value
    return 1 - (mean_absolute_error(true_levels, pred_levels) / max_len)


def mc_agreement(graph1: ag.Graph, graph2: ag.Graph) -> float:
    mc1 = graph1.major_claim or graph1.root_node
    mc2 = graph2.major_claim or graph2.root_node

    return 1.0 if mc1 == mc2 else 0.0


def _build_hierarchy(g: ag.Graph) -> list[int]:
    start = g.major_claim or g.root_node
    assert start is not None

    hierarchy: list[set[ag.Node]] = [{start}]

    # As long as the last level of the hierarchy contains at least one element
    while hierarchy[-1]:
        # Add a new level for storing the incoming nodes
        hierarchy.append(set())

        # Iterate over the nodes in the previous (-2) level and add all incoming ones the the current (-1) level
        for node in hierarchy[-2]:
            hierarchy[-1].update(g.incoming_nodes(node))

    # We do not return the first level (major claim), thus there should be at least one other level
    assert len(hierarchy) > 1

    return [len(level) for level in hierarchy[1:]]


def _normalization(graph1: ag.Graph, graph2: ag.Graph) -> float:
    return len(graph1.nodes) + len(graph2.nodes) + len(graph1.edges) + len(graph2.edges)


FUNCTIONS = [
    edit_graphkit_learn,
    # jaccard_nodes,
    jaccard_edges,
    # edit_graphmatch,
    visual_hierarchy,
    # edit_networkx,
    mc_agreement,
]
