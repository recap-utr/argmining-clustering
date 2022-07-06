import typing as t

import arguebuf

from argmining_clustering.algs.model import Result


def argument_graph(
    atoms: t.Mapping[str, arguebuf.AtomNode],
    index2id: t.Mapping[int, str],
    clustering: Result,
) -> arguebuf.Graph:
    graph = arguebuf.Graph()

    major_claim_id = index2id[clustering.mc]
    graph.add_node(atoms[major_claim_id])
    graph.major_claim = major_claim_id

    for relation in clustering.relations:
        premise_id = index2id[relation.premise]
        claim_id = index2id[relation.claim]
        scheme = arguebuf.SchemeNode()

        graph.add_edge(arguebuf.Edge(atoms[premise_id], scheme))
        graph.add_edge(arguebuf.Edge(scheme, atoms[claim_id]))

    return graph
