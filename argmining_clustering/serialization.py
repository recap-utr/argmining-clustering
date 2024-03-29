import itertools
import typing as t
from pathlib import Path

import arguebuf


def load(
    input_folder: Path, input_patterns: t.Iterable[str]
) -> t.Dict[Path, arguebuf.Graph]:
    paths = itertools.chain.from_iterable(
        input_folder.glob(pattern) for pattern in input_patterns
    )

    return {
        input_file.relative_to(input_folder): arguebuf.Graph.from_file(input_file)
        for input_file in sorted(paths)
    }


def save_json(graph: arguebuf.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_file(path.with_suffix(".json"))


def save_pdf(graph: arguebuf.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arguebuf.render(graph.to_gv(), path.with_suffix(".pdf"))
