import typing as t

from argmining_clustering.algs.model import Relation, Result


def constraint_holds(target, source, k):
    return abs(target - source) <= k


def run(MC, similarity_matrix, docs, k=2) -> Result:
    if not MC:
        MC = t.cast(int, similarity_matrix.sum(axis=1).argmax())

    order = list(range(len(docs)))
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
