import typing as t
from pathlib import Path

from typer import Typer

from argmining_clustering import evaluation, features, reconstruction, serialization
from argmining_clustering.runner import Runner

app = Typer()
PRESET_MC = False


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
        atom_nodes = list(original_graph.atom_nodes.values())

        runner = Runner(atom_nodes, mc_index)
        clustering = runner.recursive()

        reconstructed_graph = reconstruction.argument_graph(
            original_graph.atom_nodes, index2id, clustering
        )
        serialization.save(reconstructed_graph, output_folder / path, render=True)

        # For now, we do not consider the schemes between atom nodes during the evaluation
        original_graph.strip_snodes()
        reconstructed_graph.strip_snodes()

        print(evaluation.jaccard(original_graph, reconstructed_graph))


if __name__ == "__main__":
    app()
