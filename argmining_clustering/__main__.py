import typing as t
from pathlib import Path

from typer import Typer

from argmining_clustering import (
    algs,
    evaluation,
    features,
    reconstruction,
    serialization,
)

app = Typer()
PRESET_MC = false


@app.command()
def run(
    input_pattern: str,
    input_folder: Path = Path("data", "input"),
    output_folder: Path = Path("data", "output"),
):
    for path, original_graph in serialization.load(input_folder, input_pattern).items():
        index2id = dict(enumerate(original_graph.atom_nodes))
        id2index = {v: k for k, v in index2id.items()}

        assert original_graph.major_claim is not None
        mc_index = id2index[original_graph.major_claim.id]

        atom_docs = features.nlp(
            [node.plain_text for node in original_graph.atom_nodes.values()]
        )
        atom_embeddings = [features.extract_embeddings(doc) for doc in atom_docs]
        sim_matrix = features.compute_similarity_matrix(atom_embeddings)

        mc_id, relations = algs.recursive(
            dict(enumerate(atom_embeddings)),
            atom_embeddings[mc_index] if PRESET_MC else None,
        )

        reconstructed_graph = reconstruction.argument_graph(
            original_graph.atom_nodes, index2id, mc_id, relations
        )
        serialization.save(reconstructed_graph, output_folder / path, render=True)

        # For now, we do not consider the schemes between atom nodes during the evaluation
        original_graph.strip_snodes()
        reconstructed_graph.strip_snodes()

        print(evaluation.jaccard(original_graph, reconstructed_graph))


if __name__ == "__main__":
    app()
