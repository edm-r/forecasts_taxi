"""
scripts/execute_notebook_json.py
--------------------------------
Exécuteur minimal de notebooks JSON sans dépendance Jupyter.

Utile pour lancer des notebooks de benchmark dans des environnements où
`nbclient` / `jupyter` ne sont pas installés.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécute un notebook .ipynb et écrit une copie exécutée."
    )
    parser.add_argument("notebook", help="Chemin du notebook source.")
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de sortie du notebook exécuté. Défaut: <name>.executed.ipynb",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook)
    output_path = (
        Path(args.output)
        if args.output
        else notebook_path.with_name(f"{notebook_path.stem}.executed{notebook_path.suffix}")
    )

    notebook = json.loads(notebook_path.read_text())
    globals_dict: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(notebook_path.resolve()),
    }

    execution_count = 1
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = _normalize_cell_source(cell.get("source", []))
        cell["execution_count"] = execution_count
        execution_count += 1
        cell["outputs"] = []

        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            module = ast.parse(source, filename=f"{notebook_path}#cell{index}")
            body = module.body
            tail_expr = None
            if body and isinstance(body[-1], ast.Expr):
                tail_expr = ast.Expression(body[-1].value)
                module = ast.Module(body=body[:-1], type_ignores=[])

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                if module.body:
                    exec(
                        compile(module, f"{notebook_path}#cell{index}", "exec"),
                        globals_dict,
                    )
                result = None
                if tail_expr is not None:
                    result = eval(
                        compile(tail_expr, f"{notebook_path}#cell{index}", "eval"),
                        globals_dict,
                    )

            _append_stream_output(cell, "stdout", stdout.getvalue())
            _append_stream_output(cell, "stderr", stderr.getvalue())
            if tail_expr is not None and result is not None:
                cell["outputs"].append(
                    {
                        "output_type": "execute_result",
                        "data": {
                            "text/plain": repr(result).splitlines(keepends=True)
                            or [repr(result)]
                        },
                        "metadata": {},
                        "execution_count": cell["execution_count"],
                    }
                )
        except Exception:
            _append_stream_output(cell, "stdout", stdout.getvalue())
            _append_stream_output(cell, "stderr", stderr.getvalue())
            cell["outputs"].append(
                {
                    "output_type": "error",
                    "ename": "ExecutionError",
                    "evalue": "Cell execution failed",
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            output_path.write_text(json.dumps(notebook, indent=1))
            raise

    output_path.write_text(json.dumps(notebook, indent=1))
    print(output_path)


def _normalize_cell_source(source: list[str] | str) -> str:
    if isinstance(source, list):
        return "".join(source)
    return source


def _append_stream_output(cell: dict[str, Any], name: str, text: str) -> None:
    if not text:
        return
    cell["outputs"].append(
        {
            "name": name,
            "output_type": "stream",
            "text": text.splitlines(keepends=True),
        }
    )


if __name__ == "__main__":
    main()
