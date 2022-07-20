import typing as t
from dataclasses import dataclass, field
from statistics import mean

import arguebuf as ag
import graphmatch as gm
import networkx as nx
from gklearn.ged.env import GEDEnv
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

NX_OPT = {"atom_attrs": {"label": lambda x: x.id}}


def avg(graphs: t.List[t.Tuple[ag.Graph, ag.Graph]]) -> dict[str, float]:
    return {
        func.__name__: mean(func(graph1, graph2) for graph1, graph2 in graphs)
        for func in FUNCTIONS
    }


# def __str__(self):
#     return "\n".join(f"{key}: {value}" for key, value in self.run().items())


def edit_ged(graph1: ag.Graph, graph2: ag.Graph) -> float:
    """https://github.com/jajupmochi/graphkit-learn/blob/master/gklearn/examples/ged/compute_graph_edit_distance.py"""

    ged_env = GEDEnv()
    ged_env.set_edit_cost("CONSTANT", edit_cost_constants=[1, 1, 1, 1, 1, 1])
    ged_env.add_nx_graph(graph1.to_nx(**NX_OPT), "")
    ged_env.add_nx_graph(graph2.to_nx(**NX_OPT), "")

    listID = ged_env.get_all_graph_ids()
    ged_env.init(init_type="LAZY_WITHOUT_SHUFFLED_COPIES")  # type: ignore
    options = {
        "initialization_method": "RANDOM",
        "threads": 1,
    }
    ged_env.set_method("BIPARTITE", options)  # type: ignore
    ged_env.init_method()

    ged_env.run_method(listID[0], listID[1])

    pi_forward = ged_env.get_forward_map(listID[0], listID[1])
    pi_backward = ged_env.get_backward_map(listID[0], listID[1])
    dis = ged_env.get_upper_bound(listID[0], listID[1])
    print(pi_forward)
    print(pi_backward)
    print(dis)

    return dis


def edit_gm(graph1: ag.Graph, graph2: ag.Graph) -> float:
    nx1 = graph1.to_nx(**NX_OPT)
    nx2 = graph2.to_nx(**NX_OPT)

    ged = gm.GraphEditDistance(1, 1, 1, 1)
    ged.set_attr_graph_used("label", None)
    result = ged.compare([nx1, nx2], None)

    return ged.similarity(result)


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

    return jaccard_score(
        mlb.fit_transform(incoming_nodes_1),
        mlb.fit_transform(incoming_nodes_2),
        average="macro",
        zero_division=1,  # type: ignore
    )


FUNCTIONS = [jaccard, edit_gm]
