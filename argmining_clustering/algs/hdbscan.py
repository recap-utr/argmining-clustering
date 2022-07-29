import itertools
import typing as t
from collections import defaultdict
from re import L
from statistics import mean

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from argmining_clustering.algs.model import Relation, Relations, Result
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.preprocessing import normalize

from hdbscan import HDBSCAN


def run(sim_matrix: npt.NDArray[np.float_], mc_index: t.Optional[int]) -> Result:
    num_adus = sim_matrix.shape[0]
    relations: list[Relation] = []

    clusterer = HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1)
    clusterer.fit(1.0 - sim_matrix)
    g = clusterer.condensed_tree_.to_networkx()
    # Format:
    # [(10, 9), (10, 5), (10, 11), (10, 12), (11, 0), (11, 4), (11, 6), (11, 7), (11, 3), (11, 2), (12, 8), (12, 1)]
    # [(CLUSTER_ID, ADU_ID_OR_CLUSTER_ID), ...]

    clusters: defaultdict[int, set[int]] = defaultdict(set)
    cluster_connections: dict[int, int] = {}
    cluster_claims: dict[int, int] = {}

    for edge in g.edges():
        if edge[1] < num_adus:
            clusters[edge[0]].add(edge[1])
        else:
            cluster_connections[edge[1]] = edge[0]

    for cluster_id, adus in clusters.items():
        claim: int

        if mc_index and mc_index in adus:
            claim = mc_index
        else:
            similarities: defaultdict[int, list[float]] = defaultdict(list)

            for claim, premise in itertools.product(adus, adus):
                similarities[claim].append(sim_matrix[claim, premise])

            claim, _ = max(similarities.items(), key=lambda x: mean(x[1]))

            if cluster_id == min(clusters.keys()):
                mc_index = claim

        for adu in adus:
            if adu != claim:
                relations.append(Relation(adu, claim))

        if cluster_id not in cluster_claims:
            cluster_claims[cluster_id] = claim

    for cluster_target, cluster_source in cluster_connections.items():
        relations.append(
            Relation(cluster_claims[cluster_source], cluster_claims[cluster_target])
        )

    # print(clusterer.condensed_tree_.to_networkx().edges())
    # plt.figure()
    # nx.draw(g, graphviz_layout(g, prog="dot"), arrows=True)
    # plt.savefig(
    #     "./data/output/" + pendulum.now().format("YYYY-MM-DD-HH-mm-ss") + ".pdf"
    # )

    assert mc_index is not None

    return Result(mc_index, relations)
