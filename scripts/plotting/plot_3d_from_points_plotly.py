#!/usr/bin/env python3
"""
Interactive 3D scatter plot (Plotly) for (CTnCTR, PCS, MCTD) points.

Reads a CSV with columns:
  set,song,kind,CTnCTR,PCS,MCTD

Where:
  - set in {diatonic, chromatic, atonal}
  - kind in {ground_truth, computed_avg}

Marker rules:
  - ground_truth: solid circles
  - computed_avg: hollow circles

Usage:
  python plot_3d_from_points_plotly.py points_3d.csv
  python plot_3d_from_points_plotly.py points_3d.csv --label
  python plot_3d_from_points_plotly.py points_3d.csv --out plot.html
"""

import argparse
import pandas as pd
import plotly.graph_objects as go


def build_figure(df: pd.DataFrame, label: bool = False) -> go.Figure:
    # Customize colors here (your color1/2/3)
    colors = {
        "diatonic": "royalblue",   # color1
        "chromatic": "darkorange", # color2
        "atonal": "seagreen",      # color3
    }

    fig = go.Figure()

    # Stable ordering in legend
    set_order = ["diatonic", "chromatic", "atonal"]
    kind_order = ["ground_truth", "computed_avg"]

    for s in set_order:
        for k in kind_order:
            sub = df[(df["set"] == s) & (df["kind"] == k)].copy()
            if sub.empty:
                continue

            is_gt = (k == "ground_truth")
            symbol = "circle" if is_gt else "circle-open"

            # Text labels (optional)
            text = sub["song"] if label else None
            hover = (
                "<b>%{customdata[0]}</b><br>"
                "set=%{customdata[1]}<br>"
                "kind=%{customdata[2]}<br>"
                "CTnCTR=%{x}<br>"
                "PCS=%{y}<br>"
                "MCTD=%{z}<extra></extra>"
            )

            fig.add_trace(
                go.Scatter3d(
                    x=sub["CTnCTR"],
                    y=sub["PCS"],
                    z=sub["MCTD"],
                    mode="markers+text" if label else "markers",
                    text=text,
                    textposition="top center",
                    marker=dict(
                        size=6,
                        symbol=symbol,
                        color=colors.get(s, "gray"),
                        line=dict(width=2, color=colors.get(s, "gray")),
                    ),
                    name=f"{s} {'ground truth' if is_gt else 'computed avg'}",
                    customdata=sub[["song", "set", "kind"]].to_numpy(),
                    hovertemplate=hover,
                )
            )

    fig.update_layout(
        title="Ground truth vs computed average in 3D (CTnCTR, PCS, MCTD)",
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis_title="CTnCTR",
            yaxis_title="PCS",
            zaxis_title="MCTD",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to points_3d.csv")
    ap.add_argument("--label", action="store_true", help="Show song id labels next to points")
    ap.add_argument("--out", default="", help="If set, write an interactive HTML file to this path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {"set", "song", "kind", "CTnCTR", "PCS", "MCTD"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    fig = build_figure(df, label=args.label)

    if args.out:
        fig.write_html(args.out, include_plotlyjs="cdn")
        print(f"Wrote: {args.out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
