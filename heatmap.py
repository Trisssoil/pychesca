#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd
import seaborn as sns # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a continuous CHESCA correlation heatmap PDF (cluster/headless friendly)."
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
    return parser


def main() -> int:
    matplotlib.use("Agg")
    args = build_parser().parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / (Path(args.file).stem + ".pdf")

    df = pd.read_csv(args.file, index_col="RESI")
    corr = df.T.corr().abs().fillna(0)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.2,
        cbar_kws={"label": "|Correlation|"},
        ax=ax,
    )
    ax.set_title("CHESCA Continuous Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(str(out_pdf))

    print(f"Saved: {out_pdf}")
    print("Color scheme: seaborn 'mako' (perceptually uniform sequential colormap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
