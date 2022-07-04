from statistics import mean

import numpy as np
from arguebuf import AtomNode, Edge, Graph, SchemeNode
from scipy.spatial.distance import cosine


def add_branch(graph, node_source, node_target):
    node_scheme = SchemeNode()
    graph.add_edge(Edge(source=node_source, target=node_scheme))
    graph.add_edge(Edge(source=node_scheme, target=node_target))


def constraint_holds(target, source, k):

    if abs(target - source) <= k:
        return True
    else:
        return False


def construct_from_order(MC, similarity_matrix, docs, k=2):

    order = [i for i in range(len(docs))]

    nodes = [AtomNode(docs[i].text) for i in range(len(docs))]

    graph = Graph("Test")

    connected = [MC]

    unconnected = order
    unconnected.remove(MC)

    pairs = []
    while unconnected != []:

        for UC in unconnected:

            pairs = []
            for C in connected:

                if constraint_holds(C, UC, k):

                    sim = similarity_matrix[UC, C]
                    pairs.append((C, sim))

            if pairs == []:
                continue
            else:
                pairs = sorted(pairs, key=lambda tupl: tupl[1], reverse=True)

                target_position = pairs[0][0]

                unconnected.remove(UC)

                connected.append(UC)

                target = nodes[target_position]

                source = nodes[UC]

                add_branch(graph, source, target)

                break

    return graph


def compute_distances_between_members(cluster_i, cluster_j, similarity_matrix):
    pairs = []

    for member_i in cluster_i:
        for member_j in cluster_j:
            if member_i == member_j:
                continue
            else:
                similarity = similarity_matrix[member_i, member_j]
                pairs.append((member_i, member_j, similarity))

    # return pairs sorted in descending order
    return pairs


def compute_distance_between_centroids(cluster_i, cluster_j, docs):

    centroid_i = np.mean([docs[member_i].vector for member_i in cluster_i])
    centroid_j = np.mean([docs[member_j].vector for member_j in cluster_j])

    sim = +1 - cosine(centroid_i, centroid_j)
    return sim


def compute_distances_between_clusters(clusters, similarity_matrix, mode, docs):

    traversed = []
    results = {}
    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters):
            if cluster_i == cluster_j:
                continue

            # (0,1) and (1,0) return the same score - thus skip
            if (cluster_i, cluster_j) in traversed or (
                cluster_j,
                cluster_i,
            ) in traversed:
                continue

            pairs = compute_distances_between_members(
                cluster_i, cluster_j, similarity_matrix
            )

            if mode == "single":  # smallest distance between members of

                pairs = sorted(pairs, key=lambda tupl: tupl[2], reverse=False)
                results[(i, j)] = pairs[0][2]

            if mode == "complete":  # distances between all members of all clusters

                pairs = sorted(pairs, key=lambda tupl: tupl[2], reverse=True)
                results[(i, j)] = pairs[0][2]

            if mode == "average":  # largest distance between members of

                similarities = [similarity for (_, _, similarity) in pairs]
                results[(i, j)] = mean(similarities)

            if mode == "centroid":

                results[(i, j)] = compute_distance_between_centroids(
                    cluster_i, cluster_j, docs
                )

            if mode == None:
                raise Exception("Noob")

            traversed.append((cluster_i, cluster_j))

    return results


def next_clusters(clusters, similarity_matrix, mode, docs):

    scores = compute_distances_between_clusters(clusters, similarity_matrix, mode, docs)
    i, j = max(scores, key=scores.get)

    # use cluster_i to store old clusters and new clusters to know what happend in this step
    # so algorithm can decide what to do

    return i, j


def identify_target_node(cluster, query, similarity_matrix):

    candidates = []

    for member in cluster:

        similarity = similarity_matrix[member, query]

        candidates.append((member, similarity))

    candidates = sorted(candidates, key=lambda tupl: tupl[1], reverse=True)

    position_of_most_similar = candidates[0][0]

    return position_of_most_similar


# UNKNOWN MAJOR CLAIM
def construct_from_clustering(clusters, similarity_matrix, docs, mode="average"):

    nodes = [AtomNode(docs[i].text) for i in range(len(docs))]
    anker = [cluster[0] for cluster in clusters]

    graph = Graph("Test")

    # THIS ENABLES to start the clustering process with 2 elements per cluster if needed
    # [A,B] order is enforced for first branch connection from A (target) to B (source)
    for cluster in clusters:

        if len(cluster) == 2:
            source = nodes[cluster[1]]
            target = nodes[cluster[0]]
            add_branch(graph, source, target)

    while len(clusters) > 1:

        i, j = next_clusters(clusters, similarity_matrix, mode, docs)

        cluster_i, cluster_j = (clusters[i], clusters[j])
        anker_i, anker_j = (anker[i], anker[j])

        merged_anker = anker_i if len(cluster_i) >= len(cluster_j) else anker_j
        merged_cluster = clusters[i] + clusters[j]

        if len(cluster_i) >= len(cluster_j):
            larger_cluster = cluster_i
            source_anker = anker_j
        else:
            larger_cluster = cluster_j
            source_anker = anker_i

        target_position = identify_target_node(
            cluster=larger_cluster,
            query=source_anker,
            similarity_matrix=similarity_matrix,
        )

        target = nodes[target_position]

        source = nodes[source_anker]

        add_branch(graph, source, target)

        del clusters[j]
        del anker[j]
        del clusters[i]
        del anker[i]

        clusters.append(merged_cluster)
        anker.append(merged_anker)

    return graph


def construct_from_clustering_with_MC_available(
    clusters, MC, similarity_matrix, docs, mode="average"
):

    nodes = [AtomNode(docs[i].text) for i in range(len(docs))]
    anker = [cluster[0] for cluster in clusters]
    majorclaim_in = [True if MC in cluster else False for cluster in clusters]

    graph = Graph("Test")

    # THIS ENABLES to start the clustering process with 2 elements per cluster if needed
    # [A,B] order is enforced for first branch connection from A (target) to B (source)
    for cluster in clusters:

        if len(cluster) == 2:
            source = nodes[cluster[1]]
            target = nodes[cluster[0]]
            add_branch(graph, source, target)

    while len(clusters) > 1:

        i, j = next_clusters(clusters, similarity_matrix, mode, docs)

        cluster_i, cluster_j = (clusters[i], clusters[j])
        anker_i, anker_j = (anker[i], anker[j])

        ######### check if MJ in cluster i and j
        # if majorclaim_in[i] then change bahvior else normal behavior

        # merged_anker = anker_i if len(cluster_i) >= len(cluster_j) else anker_j
        new_cluster = clusters[i] + clusters[j]

        ######### make MC anker
        # if majorclaim_in[i] then mke sure its anker
        if majorclaim_in[i] or majorclaim_in[j]:

            if majorclaim_in[i]:
                main_cluster = cluster_i
                source_anker = anker_j
                new_anker = anker_i

            else:
                main_cluster = cluster_j
                source_anker = anker_i
                new_anker = anker_j

            target_position = MC

        else:
            if len(cluster_i) >= len(cluster_j):
                larger_cluster = cluster_i
                source_anker = anker_j
                new_anker = anker_i
            else:
                larger_cluster = cluster_j
                source_anker = anker_i
                new_anker = anker_j

            target_position = identify_target_node(
                cluster=larger_cluster,
                query=source_anker,
                similarity_matrix=similarity_matrix,
            )

        target = nodes[target_position]

        source = nodes[source_anker]

        add_branch(graph, source, target)

        majorclaim_in.append(True if majorclaim_in[i] or majorclaim_in[j] else False)

        del clusters[j]
        del anker[j]
        del majorclaim_in[j]
        del clusters[i]
        del anker[i]
        del majorclaim_in[i]

        clusters.append(new_cluster)
        anker.append(new_anker)

    return graph
