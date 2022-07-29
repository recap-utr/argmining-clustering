import typing as t
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import time

import arguebuf as ag
import typer
from rich import print
from rich.progress import track
from rich.table import Table

from argmining_clustering import evaluation, features, reconstruction, serialization
from argmining_clustering.runner import Runner

app = typer.Typer()


@app.command()
def run(
    input_patterns: t.List[str],
    input_folder: Path = Path("data", "input"),
    output_folder: t.Optional[Path] = None,
    predict_mc: bool = False,
    invert_sim: bool = False,
    model: str = "en_core_web_lg",
):
    features.load_spacy(model)
    global_eval: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cases = serialization.load(input_folder, input_patterns)

    for path, original_graph in track(cases.items()):
        # For now, we do not consider the schemes between atom nodes during the evaluation
        original_stripped_graph = original_graph.copy().strip_scheme_nodes()

        index2id = dict(enumerate(original_graph.atom_nodes))
        id2index = {v: k for k, v in index2id.items()}

        mc = original_graph.major_claim or original_graph.root_node
        assert mc is not None

        mc_index = None if predict_mc else id2index[mc.id]
        atom_nodes = list(original_graph.atom_nodes.values())

        runner = Runner(atom_nodes, mc_index, invert_sim)
        local_eval: defaultdict[str, list[tuple[ag.Graph, ag.Graph]]] = defaultdict(
            list
        )

        for method_name, method in runner.methods.items():
            start = time()
            clustering = method()
            end = time()
            reconstructed_graph = reconstruction.argument_graph(
                original_graph.atom_nodes, index2id, clustering
            )

            if output_folder:
                serialization.save(
                    reconstructed_graph, output_folder / path / method_name, render=True
                )

            reconstructed_graph.strip_scheme_nodes()
            local_eval[method_name].append(
                (original_stripped_graph, reconstructed_graph)
            )
            global_eval[method_name]["duration"].append(
                (end - start) * 1000
            )  # store in ms

        for clustering_name, values in local_eval.items():
            avg = evaluation.avg(clustering_name, values)

            for eval_func_name, eval_func_value in avg.items():
                global_eval[clustering_name][eval_func_name].append(eval_func_value)

    funcs = ["duration"] + sorted(
        [func_name for func_name in evaluation.FUNCTIONS.keys()]
    )

    for aggregation in (mean, min, max, median):
        table = Table("method", *funcs, title=aggregation.__name__)

        for clustering_name, eval in global_eval.items():
            values = ["{:.3f}".format(aggregation(eval[func])) for func in funcs]
            table.add_row(clustering_name, *values)

        print(table)


if __name__ == "__main__":
    app()
