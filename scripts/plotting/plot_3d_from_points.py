# #!/usr/bin/env python3
# """Plot 3D points (CTnCTR, PCS, MCTD) for ground truth vs computed averages."""
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("csv_path", help="Path to CSV (set,song,kind,CTnCTR,PCS,MCTD)")
#     parser.add_argument("--label", action="store_true", help="Annotate points with song id")
#     args = parser.parse_args()

#     # --- Global font sizing (applies to title, labels, ticks, legend by default) ---
#     plt.rcParams.update({
#         "font.size": 14,        # base font
#         "axes.titlesize": 16,
#         "axes.labelsize": 16,
#         "xtick.labelsize": 11,
#         "ytick.labelsize": 11,
#         "legend.fontsize": 12,
#     })

#     df = pd.read_csv(args.csv_path)

#     colors = {
#         "diatonic": "tab:blue",
#         "chromatic": "tab:orange",
#         "atonal": "tab:green",
#     }

#     # --- 3:2 aspect ratio via figure size ---
#     fig = plt.figure(figsize=(9, 6), dpi=150)
#     ax = fig.add_subplot(111, projection="3d")

#     # Optional: make the 3D axes box aspect nicer (Matplotlib 3.3+)
#     # Choose what looks best for your data; (1,1,1) is cube-like, (3,2,2) is wider.
#     try:
#         ax.set_box_aspect((3, 2, 2))
#     except Exception:
#         pass  # older Matplotlib; safe to ignore

#     for set_name, color in colors.items():
#         sub = df[df["set"] == set_name]

#         gt = sub[sub["kind"] == "ground_truth"]
#         if not gt.empty:
#             ax.scatter(
#                 gt["CTnCTR"], gt["PCS"], gt["MCTD"],
#                 marker="o",
#                 s=80,  # bigger markers
#                 facecolors=color,
#                 edgecolors=color,
#                 label=f"{set_name} - Human",
#             )
#             if args.label:
#                 for _, r in gt.iterrows():
#                     ax.text(r["CTnCTR"], r["PCS"], r["MCTD"], str(r["song"]), fontsize=10)

#         avg = sub[sub["kind"] == "computed_avg"]
#         if not avg.empty:
#             ax.scatter(
#                 avg["CTnCTR"], avg["PCS"], avg["MCTD"],
#                 marker="o",
#                 s=80,
#                 facecolors="none",
#                 edgecolors=color,
#                 linewidths=2.0,
#                 label=f"{set_name} - ECHO",
#             )
#             if args.label:
#                 for _, r in avg.iterrows():
#                     ax.text(r["CTnCTR"], r["PCS"], r["MCTD"], str(r["song"]), fontsize=10)

#     ax.set_xlabel("CTnCTR", labelpad=10)
#     ax.set_ylabel("PCS", labelpad=10)
#     ax.set_zlabel("MCTD", labelpad=10)
#     ax.set_title("ECHO vs Human Composed Harmonizations", pad=18)

#     # Legend sizing is controlled by rcParams above; you can also force it here:
#     ax.legend(loc="best")

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
3D scatter: ECHO (mean) vs human-composed (ground truth) for CTnCTR, PCS, MCTD.

Expected columns (like your file):
  Style,
  CTnCTR_mean, PCS_mean, MCTD_mean,
  CTnCTR_gt,   PCS_gt,   MCTD_gt
Optional label column: File (or song / id)
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def pick_label_col(df: pd.DataFrame) -> str | None:
    for c in ["File", "song", "id", "name"]:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to CSV")
    parser.add_argument("--label", action="store_true", help="Annotate points with File/id")
    args = parser.parse_args()

    # --- font sizing ---
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 16,
    })

    df = pd.read_csv(args.csv_path)

    required = [
        "Style",
        "CTnCTR_mean", "PCS_mean", "MCTD_mean",
        "CTnCTR_gt",   "PCS_gt",   "MCTD_gt",
    ]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    label_col = pick_label_col(df)

    # Assign colors from Matplotlib's default cycle (no hard-coded colors)
    styles = list(pd.unique(df["Style"]))
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    style_to_color = {s: cycle[i % len(cycle)] for i, s in enumerate(styles)}

    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    try:
        ax.set_box_aspect((3, 2, 2))
    except Exception:
        pass

    for style in styles:
        sub = df[df["Style"] == style]
        color = style_to_color[style]

        # Human (ground truth): filled markers
        ax.scatter(
            sub["CTnCTR_gt"], sub["PCS_gt"], sub["MCTD_gt"],
            marker="o", s=80,
            facecolors=color, edgecolors=color,
            label=f"{style} - Human",
        )

        # ECHO (mean): hollow markers
        ax.scatter(
            sub["CTnCTR_mean"], sub["PCS_mean"], sub["MCTD_mean"],
            marker="o", s=80,
            facecolors="none", edgecolors=color,
            linewidths=2.0,
            label=f"{style} - ECHO",
        )

        if args.label and label_col is not None:
            for _, r in sub.iterrows():
                # label near the ECHO point to reduce clutter
                lab = r[label_col]
                if isinstance(lab, str) and label_col == "File":
                    lab = os.path.splitext(os.path.basename(lab))[0]
                ax.text(r["CTnCTR_mean"], r["PCS_mean"], r["MCTD_mean"], str(lab), fontsize=9)

    ax.set_xlabel("CTnCTR", labelpad=10)
    ax.set_ylabel("PCS", labelpad=10)
    ax.set_zlabel("MCTD", labelpad=10)
    fig.suptitle("ECHO vs Human Composed", y=0.95)

    # Put legend to the right, outside the axes
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),   # (x,y) in axes fraction; x>1 moves it outside
        borderaxespad=0.0,
    )

    # Make room on the right for the legend
    fig.subplots_adjust(right=0.86, top=0.92)   # smaller -> more space for legend

    plt.show()


if __name__ == "__main__":
    main()
