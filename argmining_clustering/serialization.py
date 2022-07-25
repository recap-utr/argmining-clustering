import typing as t
from pathlib import Path

import arguebuf


def load(input_folder: Path, input_pattern: str) -> t.Dict[Path, arguebuf.Graph]:
    return {
        input_file.relative_to(input_folder): arguebuf.Graph.from_file(input_file)
        for input_file in sorted(input_folder.glob(input_pattern))
    }


def save(graph: arguebuf.Graph, path: Path, render: bool) -> None:
    graph.to_file(path.with_suffix(".json"))
    arguebuf.render(graph.to_gv(), path.with_suffix(".pdf"))
