import typing as t

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.preprocessing import normalize

MAX_LEAF_NODES = 3


def run(
    nodes: t.Mapping[int, np.ndarray], centroid: t.Optional[np.ndarray] = None
) -> t.Tuple[int, t.List[t.Tuple[int, int]]]:
    node_features = normalize(np.array(list(nodes.values())))

    if centroid is None:
        centroid = np.mean(node_features, axis=0)

    relations: t.List[t.Tuple[int, int]] = []

    node_ids = list(nodes.keys())
    claim_index, _ = pairwise_distances_argmin_min([centroid], node_features)
    claim_id = node_ids[claim_index[0]]
    premise_ids = [id for id in nodes.keys() if id != claim_id]

    if len(premise_ids) <= MAX_LEAF_NODES:
        return claim_id, [(premise_id, claim_id) for premise_id in premise_ids]

    clustering: t.Dict[int, t.Any] = {}
    scores: t.Dict[int, float] = {}

    min_clusters = 2
    max_clusters = max(len(premise_ids) // 2, 3)

    for n_clusters in range(min_clusters, max_clusters):
        features = normalize(np.array([nodes[id] for id in premise_ids]))
        curr_cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        clustering[n_clusters] = curr_cluster
        scores[n_clusters] = silhouette_score(features, curr_cluster.labels_)

    best_n_clusters = max(scores.items(), key=lambda x: x[1])[0]
    best_clustering = clustering[best_n_clusters]

    for i, nested_centroid in enumerate(best_clustering.cluster_centers_):
        clustered_premise_ids = [
            id for id, label in zip(premise_ids, best_clustering.labels_) if label == i
        ]
        clustered_claim_id, clustered_relations = run(
            {id: nodes[id] for id in clustered_premise_ids}, nested_centroid
        )

        relations.extend(clustered_relations)
        relations.append((clustered_claim_id, claim_id))

    return claim_id, relations
