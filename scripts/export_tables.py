"""Export the final paper tables from a finished results directory.

Reloads every per-cell metrics JSON written by ``scripts/run_experiments.py``
(under ``results_dir/<dataset>/<ablation>/<generator>_seed<seed>.json``),
aggregates them to ``mean ± std`` across seeds, and writes three artifacts into
``--out-dir``:

* ``summary.csv`` — tidy long table (one row per dataset/ablation/generator/metric);
* ``<dataset>__<ablation>.md`` — human-readable markdown pivots;
* ``<dataset>__<ablation>.tex`` — booktabs LaTeX tables ready to drop into the
  paper's Results/Ablation sections (``\\usepackage{booktabs}``).

This runs no training or evaluation — it only reshapes results that already exist,
so it can be re-run any time the matrix is extended::

    poetry run python scripts/export_tables.py --results-dir results/matrix \
        --out-dir results/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spade.experiments import load_records, write_tables
from spade.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="directory of per-cell JSON results (the run_matrix results_dir).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to write tables (default: <results-dir>/tables).",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="skip the .tex booktabs tables (write only CSV + markdown).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "tables"

    records = load_records(results_dir)
    if not records:
        logger.warning("no per-cell results found under %s", results_dir)
    else:
        logger.info("loaded %d cell records from %s", len(records), results_dir)

    written = write_tables(records, out_dir, latex=not args.no_latex)
    for label, path in written.items():
        logger.info("wrote %s -> %s", label, path)
    logger.info("exported %d table files to %s", len(written), out_dir)


if __name__ == "__main__":
    main()
