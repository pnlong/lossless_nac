#!/usr/bin/env python3
"""Poster-style grouped bar chart from figs/table.csv compression rate comparison."""

import argparse
from os.path import dirname, join, realpath

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

METHOD_COLS = ["FLAC", "In-context", "Standard", "Trilobyte", "Transfer"]
BIT_DEPTHS = [8, 16, 24]
BIT_COL = "b (Bits)"
DOMAIN_COL = "Domain"
DATASET_COL = "Dataset"
NA_MARKER = "\u2718"  # ✘

FIGSIZE = (20, 4.444)
Y_AXIS_LABEL = "Compression Rate (x)"
X_TICK_ROTATION = 0
WSPACE = 0.1
BAR_GROUP_WIDTH = 0.8
BRACKET_HALF_WIDTH = 0.47


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="table_poster_plot",
        description="Plot compression rate comparison from table.csv",
    )
    default_dir = dirname(realpath(__file__))
    parser.add_argument(
        "--input_filepath",
        type=str,
        default=join(default_dir, "table.csv"),
        help="Absolute filepath to the input CSV file.",
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        default=join(default_dir, "table_poster_plot.pdf"),
        help="Absolute filepath to the output PDF file.",
    )
    parser.add_argument(
        "--show_values",
        action="store_true",
        help="Display compression rate on top of each bar (NA shown as ✘).",
    )
    parser.add_argument(
        "--independent_y",
        action="store_true",
        help="Use separate y-axis scales for each bit-depth panel.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def load_and_prepare(input_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(input_filepath)
    for col in METHOD_COLS:
        df[col] = pd.to_numeric(df[col].replace("NA", pd.NA), errors="coerce")
    return df


def melt_panel(panel_df: pd.DataFrame, *, drop_na: bool = True) -> pd.DataFrame:
    id_cols = [BIT_COL, DOMAIN_COL, DATASET_COL]
    long_df = panel_df.melt(
        id_vars=id_cols,
        value_vars=METHOD_COLS,
        var_name="method",
        value_name="compression_rate",
    )
    long_df["method"] = pd.Categorical(long_df["method"], categories=METHOD_COLS, ordered=True)
    if drop_na:
        long_df = long_df.dropna(subset=["compression_rate"])
    return long_df


def bar_slot_x(dataset_idx: int, hue_idx: int, n_hue: int) -> float:
    bar_w = BAR_GROUP_WIDTH / n_hue
    return dataset_idx - BAR_GROUP_WIDTH / 2 + bar_w / 2 + hue_idx * bar_w


def patch_at_slot(container, dataset_idx: int, hue_idx: int, n_hue: int):
    target_x = bar_slot_x(dataset_idx, hue_idx, n_hue)
    for patch in container.patches:
        px = patch.get_x() + patch.get_width() / 2
        if abs(px - target_x) < 0.15:
            return patch
    return None


def method_colors_from_legend(handles, labels) -> dict[str, tuple]:
    """Map method name to color from legend handles (covers methods with no bars in a panel)."""
    return {label: handle.get_facecolor() for handle, label in zip(handles, labels)}


def add_value_labels(
    ax,
    panel_df: pd.DataFrame,
    dataset_order: list[str],
    method_colors: dict[str, tuple],
) -> None:
    """Label bar tops with values (bar color) and NA slots with ✘ (method color)."""
    n_hue = len(METHOD_COLS)
    value_lookup = panel_df.set_index(DATASET_COL)
    y_na = ax.get_ylim()[0] + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for hue_idx, method in enumerate(METHOD_COLS):
        container = ax.containers[hue_idx]
        color = method_colors[method]

        for dataset_idx, dataset in enumerate(dataset_order):
            raw = value_lookup.at[dataset, method]
            patch = patch_at_slot(container, dataset_idx, hue_idx, n_hue)

            if pd.isna(raw):
                x = bar_slot_x(dataset_idx, hue_idx, n_hue)
                ax.text(x, y_na, NA_MARKER, ha="center", va="bottom", color=color, fontsize=7, clip_on=False)
            elif patch is not None:
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_height(),
                    f"{raw:.2f}",
                    ha="center",
                    va="bottom",
                    color=patch.get_facecolor(),
                    fontsize=6,
                    clip_on=False,
                )


def bracket_y_positions(ax) -> tuple[float, float, float]:
    """Return bracket_y, label_y, tick_h in axes-fraction coords below the x-axis."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    label_bottoms = []
    for label in ax.get_xticklabels():
        if not label.get_text():
            continue
        bb = label.get_window_extent(renderer)
        label_bottoms.append(inv.transform((bb.x0, bb.y0))[1])
    if not label_bottoms:
        return -0.14, -0.22, 0.04
    lowest = min(label_bottoms)
    bracket_y = lowest - 0.05
    label_y = bracket_y - 0.05
    tick_h = 0.04
    return bracket_y, label_y, tick_h


def add_domain_brackets(ax, panel_df: pd.DataFrame) -> None:
    """Draw bracket lines and domain labels beneath the x-axis for consecutive domain groups."""
    if panel_df.empty:
        return

    bracket_y, label_y, bracket_tick_h = bracket_y_positions(ax)
    transform = ax.get_xaxis_transform()
    bracket_style = dict(color="0.35", linewidth=0.8, transform=transform, clip_on=False)

    start_idx = 0
    current_domain = panel_df.iloc[0][DOMAIN_COL]
    n = len(panel_df)

    def draw_bracket(x0: int, x1: int, domain: str) -> None:
        left, right = x0 - BRACKET_HALF_WIDTH, x1 + BRACKET_HALF_WIDTH
        ax.plot([left, right], [bracket_y, bracket_y], **bracket_style)
        ax.plot([left, left], [bracket_y, bracket_y + bracket_tick_h], **bracket_style)
        ax.plot([right, right], [bracket_y, bracket_y + bracket_tick_h], **bracket_style)
        ax.text((x0 + x1) / 2, label_y, domain, ha="center", va="top", transform=transform, fontsize=8, clip_on=False)

    for i in range(1, n):
        domain = panel_df.iloc[i][DOMAIN_COL]
        if domain != current_domain:
            draw_bracket(start_idx, i - 1, current_domain)
            start_idx = i
            current_domain = domain
    draw_bracket(start_idx, n - 1, current_domain)


def style_axes(ax, *, is_leftmost: bool, independent_y: bool) -> None:
    ax.set_facecolor("none")
    show_left_spine = is_leftmost or independent_y
    sns.despine(ax=ax, top=True, right=True, left=not show_left_spine)
    ax.grid(True, axis="y")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("")
    ax.set_ylabel(Y_AXIS_LABEL if is_leftmost else "")
    if not is_leftmost and not independent_y:
        ax.tick_params(axis="y", labelleft=False, left=False)
    elif not is_leftmost and independent_y:
        ax.tick_params(axis="y", left=True)


def main() -> None:
    args = parse_args()
    df = load_and_prepare(args.input_filepath)

    panels = {b: df[df[BIT_COL] == b].copy() for b in BIT_DEPTHS}
    width_ratios = [max(1, len(panels[b])) for b in BIT_DEPTHS]

    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_alpha(0)

    gs = gridspec.GridSpec(1, len(BIT_DEPTHS), figure=fig, width_ratios=width_ratios, wspace=WSPACE)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(BIT_DEPTHS))]

    if not args.independent_y:
        for i in range(1, len(axes)):
            axes[i].sharey(axes[0])

    legend_handles = None
    legend_labels = None
    method_colors: dict[str, tuple] = {}

    for col_idx, bit_depth in enumerate(BIT_DEPTHS):
        ax = axes[col_idx]
        panel_df = panels[bit_depth]
        dataset_order = panel_df[DATASET_COL].tolist()
        long_df = melt_panel(panel_df)

        sns.barplot(
            data=long_df,
            x=DATASET_COL,
            y="compression_rate",
            hue="method",
            order=dataset_order,
            hue_order=METHOD_COLS,
            ax=ax,
            legend=(col_idx == 0),
        )

        if col_idx == 0 and ax.get_legend() is not None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            method_colors = method_colors_from_legend(legend_handles, legend_labels)
            ax.get_legend().remove()

        plt.setp(
            ax.get_xticklabels(),
            rotation=X_TICK_ROTATION,
            ha="center",
            va="top",
            rotation_mode="anchor",
        )
        ax.set_title(f"{bit_depth}-bit")
        style_axes(ax, is_leftmost=(col_idx == 0), independent_y=args.independent_y)

        if args.show_values:
            add_value_labels(ax, panel_df, dataset_order, method_colors)

        add_domain_brackets(ax, panel_df)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(METHOD_COLS),
        )

    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.26, top=0.88)

    plt.savefig(args.output_filepath, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved plot to {args.output_filepath}.")


if __name__ == "__main__":
    main()
