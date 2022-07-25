import typing as t
from dataclasses import dataclass, field
from inspect import getmembers, isfunction, ismethod

import numpy as np
import numpy.typing as npt
from arguebuf import AtomNode, Graph
from spacy.tokens.doc import Doc

from argmining_clustering import algs, features


@dataclass
class Runner:
    atom_nodes: t.List[AtomNode]
    mc: t.Optional[int]
    atom_docs: t.List[Doc] = field(init=False)
    atom_embeddings: t.List[npt.NDArray[np.float_]] = field(init=False)
    sim_matrix: npt.NDArray[np.float_] = field(init=False)
    keyword_matrix: npt.NDArray[np.float_] = field(init=False)

    def __post_init__(self) -> None:
        self.atom_docs = features.nlp([node.plain_text for node in self.atom_nodes])
        self.atom_embeddings = [
            features.extract_embeddings(doc) for doc in self.atom_docs
        ]
        self.sim_matrix = features.compute_similarity_matrix(self.atom_embeddings)

        if any(method_name.endswith("_kw") for method_name in self.method_names()):
            self.keyword_matrix = features.compute_keyword_matching_similarity_matrix(
                self.atom_docs
            )
        else:
            self.keyword_matrix = np.array([])

    @classmethod
    def method_names(cls) -> t.List[str]:
        return [name for name in dir(cls) if name.startswith("run_")]

    @property
    def methods(self) -> t.Dict[str, t.Callable[[], algs.Result]]:
        return {
            name.removeprefix("run_"): getattr(self, name)
            for name in self.method_names()
        }

    def run_agglomerative(self) -> algs.Result:
        return algs.agglomerative(self.atom_docs, self.sim_matrix, self.mc)

    # def run_agglomerative_kw(self) -> algs.Result:
    #     return algs.agglomerative(self.atom_docs, self.keyword_matrix, self.mc)

    def run_order(self) -> algs.Result:
        return algs.order(self.mc, self.sim_matrix, self.atom_docs)

    # def run_order_kw(self) -> algs.Result:
    #     return algs.order(self.mc, self.keyword_matrix, self.atom_docs)

    def run_flat(self) -> algs.Result:
        return algs.flat(self.sim_matrix, self.mc)

    def run_random(self) -> algs.Result:
        return algs.random(list(range(len(self.atom_nodes))), self.mc)

    def run_recursive(self) -> algs.Result:
        return algs.recursive(
            dict(enumerate(self.atom_embeddings)),
            self.atom_embeddings[self.mc] if self.mc else None,
        )

    # def run_sim(self) -> algs.Result:
    #     return algs.sim(self.sim_matrix, self.mc)
