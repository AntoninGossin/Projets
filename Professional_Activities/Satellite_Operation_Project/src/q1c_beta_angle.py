"""
Question 1(c): beta angle of all the satellites, combined in a single plot.

One axes, one line per satellite, over the full year of the beta file. The window
covered by the telemetry files is shaded. Satellites are named twice: in the legend
and directly at the right end of their lines. Satellites that end within MERGE_TOL
degrees of each other share one label, so the labels never collide.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from telemetry import (MUTED_COLOR, SAT_COLORS, SATELLITES, TEXT_COLOR,
                       load_beta, save, setup_style)

MERGE_TOL = 0.5      # deg: end values closer than this share one label
LABEL_GAP_PT = 12.0  # minimum vertical distance between two labels, in points
DOT_STEP_PT = 7.0    # spacing of the colored markers in front of a label
# Overlapping curves get different line styles (drawn on top of each other), so that
# every satellite stays visible and identity does not rely on color alone.
LINE_STYLES = ["-", "--", ":"]


def group_by_end_value(end_values: pd.Series, tol: float = MERGE_TOL) -> list[list[str]]:
    """Group satellites whose last value is within tol of the group's lowest value."""
    ordered = end_values.sort_values()
    groups = [[ordered.index[0]]]
    for name in ordered.index[1:]:
        if ordered[name] - ordered[groups[-1][0]] <= tol:
            groups[-1].append(name)
        else:
            groups.append([name])
    # keep the standard satellite order inside each group
    return [sorted(g, key=SATELLITES.index) for g in groups]


def spread(positions: list[float], min_gap: float) -> list[float]:
    """Push sorted positions apart until neighbours are min_gap apart, around their centre."""
    pos = list(positions)
    for _ in range(100):
        moved = False
        for i in range(1, len(pos)):
            deficit = min_gap - (pos[i] - pos[i - 1])
            if deficit > 1e-9:
                pos[i - 1] -= deficit / 2
                pos[i] += deficit / 2
                moved = True
        if not moved:
            break
    return pos


def add_end_labels(ax: plt.Axes, beta: pd.DataFrame, groups: list[list[str]]) -> None:
    """Name the lines at their right end; satellites ending together share one label."""
    fig = ax.figure
    x_end = beta.index[-1]
    y_line = [beta.iloc[-1][g].mean() for g in groups]

    # vertical room: points per degree, to keep labels at least LABEL_GAP_PT apart
    ymin, ymax = ax.get_ylim()
    pt_per_deg = ax.get_window_extent().height * 72 / fig.dpi / (ymax - ymin)
    y_text = spread(y_line, LABEL_GAP_PT / pt_per_deg)

    def shifted(dx_pt: float) -> mtransforms.Transform:
        # x in data, moved right by a few points: the labels sit just outside the axes
        return mtransforms.offset_copy(ax.transData, fig=fig, x=dx_pt, units="points")

    for names, y in zip(groups, y_text):
        for i, name in enumerate(names):  # one small colored marker per satellite
            ax.plot([x_end], [y], marker="o", markersize=4.5, color=SAT_COLORS[name],
                    linestyle="none", transform=shifted(8 + i * DOT_STEP_PT), clip_on=False)
        ax.text(x_end, y, " / ".join(names), transform=shifted(8 + len(names) * DOT_STEP_PT),
                ha="left", va="center", fontsize=9, color=TEXT_COLOR, clip_on=False)


def run(beta: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp]) -> list[Path]:
    """Plot the beta angle of every satellite and shade the telemetry window."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.07, right=0.83, top=0.85, bottom=0.09)  # room for the labels

    # axis limits: whole year, y rounded out to the next 10 degrees
    x0, x1 = beta.index[0], beta.index[-1]
    ylo = np.floor(beta.min().min() / 10) * 10
    yhi = np.ceil(beta.max().max() / 10) * 10

    ax.axhline(0, color=MUTED_COLOR, linewidth=0.8, zorder=1.5)  # thin zero line

    # telemetry window, clipped to the dates covered by the beta file
    w0, w1 = max(window[0], x0), min(window[1], x1)
    ax.axvspan(w0, w1, color=MUTED_COLOR, alpha=0.2, linewidth=0, zorder=1)
    ax.annotate(f"telemetry window\n{window[0]:%b %d %H:%M} to {window[1]:%b %d %H:%M} UTC",
                xy=(mdates.date2num(w0), 0.65), xycoords=ax.get_xaxis_transform(),  # x data, y axes
                xytext=(-26, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8.5, color=MUTED_COLOR,
                arrowprops=dict(arrowstyle="-|>", color=MUTED_COLOR, linewidth=0.8,
                                shrinkA=2, shrinkB=0))

    # satellites ending together are drawn with different styles, one above the other
    groups = group_by_end_value(beta.iloc[-1])
    lines = {}
    for names in groups:
        for k, name in enumerate(names):
            lines[name], = ax.plot(beta.index, beta[name], color=SAT_COLORS[name],
                                   linewidth=1.4, linestyle=LINE_STYLES[k % len(LINE_STYLES)],
                                   zorder=2 + k)

    ax.set_xlim(x0, x1)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(np.arange(ylo, yhi + 1, 10))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta angle [deg]")
    ax.set_title(f"Daily beta angle of the {len(SATELLITES)} satellites, "
                 f"{x0:%b %d, %Y} to {x1:%b %d, %Y}", loc="left", pad=26, color=TEXT_COLOR)
    ax.legend([lines[name] for name in SATELLITES], SATELLITES, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=len(SATELLITES), handlelength=2.4,
              columnspacing=1.6, borderaxespad=0.3)

    add_end_labels(ax, beta, groups)
    return [save(fig, "q1c_beta_angle.png")]


if __name__ == "__main__":
    start = time.perf_counter()
    setup_style()
    paths = run(load_beta(), (pd.Timestamp("2019-05-30 10:00"), pd.Timestamp("2019-06-02 10:00")))
    elapsed = time.perf_counter() - start
    for path in paths:
        print(f"wrote {path}")
    print(f"q1c done in {elapsed:.1f} s")
