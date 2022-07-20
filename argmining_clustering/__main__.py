import typing as t
from collections import defaultdict
from pathlib import Path
from statistics import mean

import arguebuf as ag
import typer

from argmining_clustering import evaluation, reconstruction, serialization
from argmining_clustering.runner import Runner

app = typer.Typer()
PRESET_MC = False


@app.command()
def run(
    input_pattern: str,
    input_folder: Path = Path("data", "input"),
    output_folder: t.Optional[Path] = None,
):
    global_eval: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path, original_graph in serialization.load(input_folder, input_pattern).items():
        # For now, we do not consider the schemes between atom nodes during the evaluation
        original_stripped_graph = original_graph.copy().strip_scheme_nodes()

        index2id = dict(enumerate(original_graph.atom_nodes))
        id2index = {v: k for k, v in index2id.items()}

        assert original_graph.major_claim is not None
        mc_index = id2index[original_graph.major_claim.id]
        atom_nodes = list(original_graph.atom_nodes.values())

        runner = Runner(atom_nodes, mc_index)
        local_eval: defaultdict[str, list[tuple[ag.Graph, ag.Graph]]] = defaultdict(
            list
        )

        for method_name, method in runner.methods.items():
            clustering = method()
            reconstructed_graph = reconstruction.argument_graph(
                original_graph.atom_nodes, index2id, clustering
            )

            if output_folder:
                serialization.save(
                    reconstructed_graph, output_folder / path, render=True
                )

            reconstructed_graph.strip_scheme_nodes()
            local_eval[method_name].append(
                (original_stripped_graph, reconstructed_graph)
            )

        for clustering_name, values in local_eval.items():
            avg = evaluation.avg(values)

            for eval_func_name, eval_func_value in avg.items():
                global_eval[clustering_name][eval_func_name].append(eval_func_value)

    for clustering_name, eval in global_eval.items():
        typer.echo(clustering_name)

        for func_name, func_values in eval.items():
            typer.echo(f"{func_name}={mean(func_values)}")


if __name__ == "__main__":
    app()
