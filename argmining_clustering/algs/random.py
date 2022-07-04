import random
import typing as t


def run(
    nodes: t.List[int], mc_index: t.Optional[int] = None
) -> t.Tuple[int, t.List[t.Tuple[int, int]]]:
    random.seed(0)

    if not mc_index:
        mc_index = random.choice(nodes)

    sampled_nodes = [mc_index]
    relations = []

    while len(nodes) > 0:
        premise = random.choice(nodes)
        claim = random.choice(sampled_nodes)
        relations.append((premise, claim))

        nodes.remove(premise)
        sampled_nodes.append(premise)

    return (mc_index, relations)
