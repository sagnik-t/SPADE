"""Aggregate per-cell results into mean±std tables across seeds.

Takes the flat records produced by :func:`spade.experiments.run_matrix`, flattens
each cell's nested metric dict into scalar columns, and summarizes them across
seeds for every (dataset, ablation, generator). Writes a machine-readable long CSV
(``summary.csv``) plus one human-readable markdown table per (dataset, ablation)
with generators as rows and ``mean±std`` cells — the format that drops into the
paper's result tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import pandas as pd

__all__ = ["load_records", "flatten_cell", "build_summary", "write_tables"]

_CELL_RE = re.compile(r"^(?P<generator>.+)_seed(?P<seed>\d+)\.json$")


def load_records(results_dir: str | Path) -> list[dict]:
    """Reload per-cell records written by :func:`spade.experiments.run_matrix`.

    Walks ``results_dir/<dataset>/<ablation>/<generator>_seed<seed>.json`` and
    rebuilds the ``{dataset, ablation, generator, seed, cell}`` records that
    :func:`build_summary` and :func:`write_tables` consume — so tables can be
    (re)generated from a finished results directory without re-running anything.
    """
    root = Path(results_dir)
    records: list[dict] = []
    for path in sorted(root.glob("*/*/*.json")):
        match = _CELL_RE.match(path.name)
        if match is None:
            continue
        ablation_dir = path.parent
        dataset_dir = ablation_dir.parent
        records.append({
            "dataset": dataset_dir.name,
            "ablation": ablation_dir.name,
            "generator": match.group("generator"),
            "seed": int(match.group("seed")),
            "cell": json.loads(path.read_text()),
        })
    return records


def flatten_cell(cell: dict) -> dict[str, float]:
    """Flatten one cell's nested metrics into scalar ``name -> value`` columns."""
    out: dict[str, float] = {}
    syn = cell.get("synthetic") or {}
    for key in ("density", "nnz"):
        if key in syn:
            out[key] = syn[key]
    geometry = cell.get("geometry")
    if geometry:
        for ref_model, metrics in geometry.items():
            for k, v in metrics.items():
                out[f"{ref_model}/{k}"] = v
    for block in ("latent", "degree", "tstr"):
        values = cell.get(block)
        if values:
            out.update(values)
    return {k: float(v) for k, v in out.items() if v is not None}


def _long_frame(records: list[dict]) -> pd.DataFrame:
    """Tidy long frame: one row per (dataset, ablation, generator, seed, metric)."""
    rows: list[dict] = []
    for rec in records:
        flat = flatten_cell(rec["cell"])
        for metric, value in flat.items():
            rows.append({
                "dataset": rec["dataset"],
                "ablation": rec["ablation"],
                "generator": rec["generator"],
                "seed": rec["seed"],
                "metric": metric,
                "value": value,
            })
    return pd.DataFrame(rows)


def build_summary(records: list[dict]) -> pd.DataFrame:
    """Mean/std/count across seeds per (dataset, ablation, generator, metric)."""
    long = _long_frame(records)
    if long.empty:
        return pd.DataFrame(
            columns=["dataset", "ablation", "generator", "metric", "mean", "std", "n_seeds"]
        )
    grouped = (
        long.groupby(["dataset", "ablation", "generator", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_seeds"})
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _markdown_table(block: pd.DataFrame) -> str:
    """One ``mean±std`` pivot (generators × metrics) as a markdown table.

    Rendered by hand (no ``tabulate`` dependency).
    """
    block = block.copy()
    block["cell"] = block.apply(
        lambda r: f"{r['mean']:.4f}±{r['std']:.4f}", axis=1
    )
    pivot = block.pivot(index="generator", columns="metric", values="cell").fillna("—")
    cols = [str(c) for c in pivot.columns]
    lines = [
        "| generator | " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * (len(cols) + 1)) + " |",
    ]
    for generator, row in pivot.iterrows():
        cells = " | ".join(str(row[c]) for c in pivot.columns)
        lines.append(f"| {generator} | {cells} |")
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    """Escape the LaTeX-special characters that appear in metric/generator names."""
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace(
        "%", r"\%"
    ).replace("&", r"\&").replace("#", r"\#")


def _latex_table(block: pd.DataFrame, dataset: str, ablation: str) -> str:
    """One ``mean±std`` pivot (generators × metrics) as a booktabs LaTeX table.

    Cells are typeset in math mode as ``$mean \\pm std$``; the table uses
    ``booktabs`` rules so it drops straight into the paper (requires
    ``\\usepackage{booktabs}``).
    """
    block = block.copy()
    block["cell"] = block.apply(
        lambda r: f"${r['mean']:.4f} \\pm {r['std']:.4f}$", axis=1
    )
    pivot = block.pivot(index="generator", columns="metric", values="cell").fillna("--")
    metrics = [str(c) for c in pivot.columns]

    col_spec = "l" + "r" * len(metrics)
    header = " & ".join(["Generator", *[_latex_escape(m) for m in metrics]]) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Results on {_latex_escape(dataset)} "
        rf"({_latex_escape(ablation)}), mean $\pm$ std over seeds.}}",
        rf"\label{{tab:{dataset}-{ablation}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for generator, row in pivot.iterrows():
        cells = " & ".join(str(row[c]) for c in pivot.columns)
        lines.append(f"{_latex_escape(str(generator))} & {cells} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_tables(
    records: list[dict], out_dir: str | Path, *, latex: bool = True
) -> dict[str, Path]:
    """Write ``summary.csv`` plus markdown (and, by default, LaTeX) tables.

    Emits one ``<dataset>__<ablation>.md`` markdown pivot per group and, when
    ``latex`` is set, a matching ``<dataset>__<ablation>.tex`` booktabs table for
    the paper. Returns a map of label -> written path. With no records, writes
    only an empty ``summary.csv`` so downstream steps have a stable artifact.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records)

    written: dict[str, Path] = {}
    csv_path = out_dir / "summary.csv"
    summary.to_csv(csv_path, index=False)
    written["summary"] = csv_path

    for key, block in summary.groupby(["dataset", "ablation"]):
        dataset, ablation = cast("tuple[str, str]", key)
        md_path = out_dir / f"{dataset}__{ablation}.md"
        header = f"# {dataset} — {ablation} (mean±std over seeds)\n\n"
        md_path.write_text(header + _markdown_table(block) + "\n")
        written[f"{dataset}/{ablation}"] = md_path
        if latex:
            tex_path = out_dir / f"{dataset}__{ablation}.tex"
            tex_path.write_text(_latex_table(block, dataset, ablation) + "\n")
            written[f"{dataset}/{ablation}/tex"] = tex_path
    return written
