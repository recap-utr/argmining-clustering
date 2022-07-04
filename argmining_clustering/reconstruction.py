import typing as t

import arguebuf


def argument_graph(
    atoms: t.Mapping[str, arguebuf.AtomNode],
    index2id: t.Mapping[int, str],
    major_claim_index: int,
    premise_claim_relations: t.Iterable[t.Tuple[int, int]],
) -> arguebuf.Graph:
    graph = arguebuf.Graph()

    major_claim_id = index2id[major_claim_index]
    graph.add_node(atoms[major_claim_id])
    graph.major_claim = major_claim_id

    for premise_index, claim_index in premise_claim_relations:
        premise_id = index2id[premise_index]
        claim_id = index2id[claim_index]
        scheme = arguebuf.SchemeNode()

        graph.add_edge(arguebuf.Edge(atoms[premise_id], scheme))
        graph.add_edge(arguebuf.Edge(scheme, atoms[claim_id]))

    return graph
