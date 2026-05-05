#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from pychesca import HAC
from pychesca.plots import plot_corr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate only CHESCA correlation matrix PDF (cluster/headless friendly)."
    )
    parser.add_argument(
        "-file",
        required=True,
        help="Input CSV with RESI as index column.",
    )
    parser.add_argument(
        "-output",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "-cutoff",
        type=float,
        default=98.0,
        help="Correlation cutoff as percent (default: 98.0).",
    )
    parser.add_argument(
        "-linkage",
        default="complete",
        choices=["complete", "single", "average", "ward"],
        help="HAC linkage method (default: complete).",
    )
    parser.add_argument(
        "--out-name",
        default="correlation_matrix.pdf",
        help="Output PDF name (default: correlation_matrix.pdf).",
    )
    return parser


def main() -> int:
    matplotlib.use("Agg")
    args = build_parser().parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / args.out_name

    df = pd.read_csv(args.file, index_col="RESI")
    hac = HAC(df, cutoff=args.cutoff, method=args.linkage)
    plot_corr(hac, cutoff=args.cutoff, save_file=str(out_pdf))

    print(f"Saved: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
