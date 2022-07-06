import typing as t
from dataclasses import dataclass, field

import numpy as np
from arguebuf import AtomNode
from spacy.tokens.doc import Doc

from argmining_clustering import algs, features


@dataclass
class Runner:
    atom_nodes: t.List[AtomNode]
    mc: t.Optional[int]
    atom_docs: t.List[Doc] = field(init=False)
    atom_embeddings: t.List[np.ndarray] = field(init=False)
    sim_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.atom_docs = features.nlp([node.plain_text for node in self.atom_nodes])
        self.atom_embeddings = [
            features.extract_embeddings(doc) for doc in self.atom_docs
        ]
        self.sim_matrix = features.compute_similarity_matrix(self.atom_embeddings)

    def agglomerative(self) -> algs.Result:
        return algs.agglomerative(self.atom_docs, self.sim_matrix, self.mc)

    def flat(self) -> algs.Result:
        return algs.flat(self.sim_matrix, self.mc)

    def order(self) -> algs.Result:
        return algs.order(self.mc, self.sim_matrix, self.atom_docs)

    def random(self) -> algs.Result:
        return algs.random(list(range(len(self.atom_nodes))), self.mc)

    def recursive(self) -> algs.Result:
        return algs.recursive(
            dict(enumerate(self.atom_embeddings)),
            self.atom_embeddings[self.mc] if self.mc else None,
        )

    def sim(self) -> algs.Result:
        return algs.sim(self.sim_matrix, self.mc)
