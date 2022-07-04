import typing as t

import arguebuf
import networkx as nx
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer


def edit_distance(graph1: arguebuf.Graph, graph2: arguebuf.Graph) -> float:
    nx1 = graph1.to_nx(atom_label=lambda x: x.id)
    nx2 = graph2.to_nx(atom_label=lambda x: x.id)

    return (
        nx.graph_edit_distance(
            nx1,
            nx2,
            node_match=lambda x, y: x["label"] == y["label"],
        )
        or 0
    )


def jaccard(graph1: arguebuf.Graph, graph2: arguebuf.Graph) -> float:
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
