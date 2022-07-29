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
    cluster_connections = nx.DiGraph()
    cluster_claims: dict[int, int] = {}

    for edge in g.edges():
        cluster_id: int = edge[0]
        adu_id: int = edge[1]

        if adu_id < num_adus:
            clusters[cluster_id].add(adu_id)
        else:
            cluster_connections.add_edge(adu_id, cluster_id)

    primary_cluster = min(clusters.keys())

    # If the major claim is given, we have to manipulate the result of the algorithm
    if mc_index is not None:
        for entry in clusters.items():
            entry[1].discard(mc_index)

        clusters[primary_cluster].add(mc_index)

        # When moving the major claim, it may happen that another cluster is empty
        for key in set(clusters.keys()):
            if len(clusters[key]) == 0:
                del clusters[key]

    for cluster_id, adus in sorted(clusters.items(), key=lambda x: x[0]):
        claim: int

        if mc_index is not None and cluster_id == primary_cluster:
            claim = mc_index
        else:
            similarities: defaultdict[int, list[float]] = defaultdict(list)

            for claim, premise in itertools.product(adus, adus):
                similarities[claim].append(sim_matrix[claim, premise])

            claim, _ = max(similarities.items(), key=lambda x: mean(x[1]))

        # Only on first run of the loop
        if mc_index is None and cluster_id == primary_cluster:
            mc_index = claim

        for adu in adus:
            if adu != claim:
                relations.append(Relation(adu, claim))

        cluster_claims[cluster_id] = claim

    # https://stackoverflow.com/a/61917979
    while not set(cluster_claims).issuperset(cluster_connections.nodes()):
        to_be_removed = filter(
            lambda x: x not in cluster_claims, cluster_connections.nodes()
        )

        for node in to_be_removed:
            parent = next(cluster_connections.predecessors(node))
            cluster_connections = nx.contracted_nodes(
                cluster_connections, parent, node, self_loops=False
            )

    for edge in cluster_connections.edges():
        source: int = edge[0]
        target: int = edge[1]
        relations.append(Relation(cluster_claims[source], cluster_claims[target]))

    # print(clusterer.condensed_tree_.to_networkx().edges())
    # plt.figure()
    # nx.draw(g, graphviz_layout(g, prog="dot"), arrows=True)
    # plt.savefig(
    #     "./data/output/" + pendulum.now().format("YYYY-MM-DD-HH-mm-ss") + ".pdf"
    # )

    # Otherwise IDEs will show warnings as this is not obvious from the code itself
    assert mc_index is not None

    return Result(mc_index, relations)
