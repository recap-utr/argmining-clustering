import random
import typing as t

from argmining_clustering.algs.model import Relation, Relations, Result


def run(nodes: t.List[int], mc_index: t.Optional[int] = None) -> Result:
    random.seed(0)

    if not mc_index:
        mc_index = random.choice(nodes)

    sampled_nodes = [mc_index]
    relations: Relations = []

    while len(nodes) > 0:
        premise = random.choice(nodes)
        claim = random.choice(sampled_nodes)
        relations.append(Relation(premise, claim))

        nodes.remove(premise)
        sampled_nodes.append(premise)

    return Result(mc_index, relations)
