import typing as t

import numpy as np
import numpy.typing as npt
from argmining_clustering.algs.model import Relation, Relations, Result
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, silhouette_score
from sklearn.preprocessing import normalize

MAX_LEAF_NODES = 3


def run(
    nodes: t.Mapping[int, npt.NDArray[np.float_]],
    centroid: t.Optional[npt.NDArray[np.float_]] = None,
) -> Result:
    node_features = np.array(list(nodes.values()))

    if centroid is None:
        centroid = np.mean(node_features, axis=0)

    relations: Relations = []

    node_ids = list(nodes.keys())
    claim_index = pairwise_distances_argmin([centroid], node_features, metric="cosine")
    claim_id = node_ids[claim_index[0]]
    premise_ids = [id for id in nodes.keys() if id != claim_id]

    if len(premise_ids) <= MAX_LEAF_NODES:
        return Result(
            claim_id, [Relation(premise_id, claim_id) for premise_id in premise_ids]
        )

    clustering: t.Dict[int, t.Any] = {}
    scores: t.Dict[int, float] = {}

    min_clusters = 2  # for '1', the silhouette score cannot be computed
    max_clusters = (
        max(len(premise_ids) // 2, min_clusters) + 1
    )  # at least min_clusters + 1 needs to be passed to range()

    for n_clusters in range(min_clusters, max_clusters):
        features = normalize(np.array([nodes[id] for id in premise_ids]))
        curr_cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        clustering[n_clusters] = curr_cluster
        scores[n_clusters] = silhouette_score(features, curr_cluster.labels_)  # type: ignore

    best_n_clusters = max(scores.items(), key=lambda x: x[1])[0]
    best_clustering = clustering[best_n_clusters]

    for i, nested_centroid in enumerate(best_clustering.cluster_centers_):
        clustered_premise_ids = [
            id for id, label in zip(premise_ids, best_clustering.labels_) if label == i
        ]
        res = run({id: nodes[id] for id in clustered_premise_ids}, nested_centroid)

        relations.extend(res.relations)
        relations.append(Relation(res.mc, claim_id))

    return Result(claim_id, relations)
