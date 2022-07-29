import typing as t

from argmining_clustering.algs.model import Relation, Result


def constraint_holds(target, source, k):
    return abs(target - source) <= k


def compute_MC_sim_order(MC, similarity_matrix, docs):
    """
    Kurzer Hack, um Problem zu lösen!
    MC: position of MC in input order, eqv. to similarity matrix indexes
    """
    sim_loc_pair = [x for x in zip(similarity_matrix[MC], list(range(len(docs))))]

    sorted_sim_pos_pairs = sorted(sim_loc_pair, key=lambda tupl: tupl[0], reverse=True)
    # del sorted_sim_pos_pairs[0] # first sim loc is 1.0 to MC, thus remove

    order = [position for (sim, position) in sorted_sim_pos_pairs]

    return order


def run(MC, similarity_matrix, docs, k=2) -> Result:
    if MC is None:
        MC = t.cast(int, similarity_matrix.sum(axis=1).argmax())

    order = compute_MC_sim_order(MC, similarity_matrix, docs)
    relations = []

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

            if not pairs:
                continue

            pairs = sorted(pairs, key=lambda tupl: tupl[1], reverse=True)
            target_position = pairs[0][0]

            unconnected.remove(UC)
            connected.append(UC)

            target = target_position
            source = UC
            relations.append(Relation(source, target))

            break

    return Result(MC, relations)
