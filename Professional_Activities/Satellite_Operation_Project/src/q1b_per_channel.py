"""
Question 1(b): one plot per telemetry channel, all satellites.

Seven satellites on one axis would be unreadable (10 Hz noisy series), so each
figure is a stack of 7 small multiples (one panel per satellite) that share both
axes: the same time window and the same y scale, so values are directly comparable.
Every sample is plotted (no smoothing, resampling or decimation) so that no peak
is hidden; the lines are only cut inside outages longer than 10 minutes.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from telemetry import (
    CHANNELS, GYRO_RATE_LIMIT_DEG_S, GYROS, LABELS, LIMIT_COLOR, MUTED_COLOR,
    SAT_COLORS, SATELLITES, TEXT_COLOR, UNITS, WHEEL_SATURATION_RPM, WHEELS,
    break_gaps, break_step_gaps, channel, format_time_axis, load_all, save, setup_style,
    telemetry_window,
)

LINE_WIDTH = 0.5
OUTAGE_LABEL_MIN = pd.Timedelta("3h")  # silences at least this long get a "no data" note


def axis_spec(col: str) -> dict:
    """Fixed y-axis settings shared by all panels of a channel family."""
    if col in GYROS:
        return dict(ylim=(-5, 5), yticks=[-4, -2, 0, 2, 4],
                    limits=[-GYRO_RATE_LIMIT_DEG_S, GYRO_RATE_LIMIT_DEG_S],
                    limit_label=f"limit ±{GYRO_RATE_LIMIT_DEG_S:g} deg/s")
    if col in WHEELS:
        # lower limit slightly below 0 so a wheel stopped at 0 rpm is not hidden on the axis line
        return dict(ylim=(-300, 8800), yticks=[0, 2000, 4000, 6000, 8000],
                    limits=[WHEEL_SATURATION_RPM],
                    limit_label=f"saturation {WHEEL_SATURATION_RPM:.0f} rpm")
    if col == "low_voltage":
        return dict(ylim=(-0.15, 1.15), yticks=[0, 1], limits=[], limit_label="")
    return dict(ylim=(6.5, 8.5), yticks=[6.5, 7.0, 7.5, 8.0, 8.5], limits=[], limit_label="")


def mark_outages(ax: plt.Axes, series: pd.Series) -> None:
    """Write "no data" in the middle of every long silence (the line is cut there)."""
    gaps = series.index.to_series().diff()
    for end, gap in gaps[gaps >= OUTAGE_LABEL_MIN].items():
        # x in data coordinates, y in axes fraction (panel centre)
        ax.text(end - gap / 2, 0.5, "no data", transform=ax.get_xaxis_transform(),
                ha="center", va="center", color=MUTED_COLOR, fontsize=9, style="italic")


def plot_channel(data: dict[str, pd.DataFrame], col: str) -> Path:
    """One figure for one channel: a panel per satellite, shared axes."""
    spec = axis_spec(col)
    step = "steps-post" if col == "low_voltage" else "default"
    x_min, x_max = telemetry_window(data)

    fig, axes = plt.subplots(len(SATELLITES), 1, sharex=True, sharey=True,
                             figsize=(14, 12), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.04, hspace=0.0)

    for ax, sat in zip(axes, SATELLITES):
        series = channel(data[sat], col)
        if series.empty:
            ax.text(0.5, 0.5, "not telemetered", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED_COLOR, fontsize=11)
        else:
            line = break_step_gaps(series) if step == "steps-post" else break_gaps(series)
            ax.plot(line.index, line.values, color=SAT_COLORS[sat],
                    linewidth=LINE_WIDTH, drawstyle=step)
            mark_outages(ax, series)
            if series.nunique() == 1:  # flat channel (e.g. a wheel stopped all window)
                unit = "" if col == "low_voltage" else f" {UNITS[col]}"
                ax.text(0.5, 0.5, f"constant {series.iloc[0]:g}{unit}",
                        transform=ax.transAxes, ha="center", va="center",
                        color=MUTED_COLOR, fontsize=11)

        for level in spec["limits"]:
            ax.axhline(level, color=LIMIT_COLOR, linestyle="--", linewidth=0.9, zorder=1.8)
        # satellite name as the panel label, in neutral text color
        ax.set_ylabel(sat, rotation=0, ha="right", va="center", labelpad=8,
                      color=TEXT_COLOR, fontsize=11, fontweight="bold")

    # same limits and ticks everywhere (sharey / sharex propagate them)
    axes[0].set_ylim(*spec["ylim"])
    axes[0].set_yticks(spec["yticks"])
    axes[0].set_xlim(x_min, x_max)

    # threshold label, written once per figure in the right margin of the top panel
    if spec["limit_label"]:
        axes[0].annotate(spec["limit_label"], xy=(1.005, spec["limits"][-1]),
                         xycoords=axes[0].get_yaxis_transform(), ha="left", va="center",
                         color=MUTED_COLOR, fontsize=8)

    format_time_axis(axes[-1])
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(f"{LABELS[col]} [{UNITS[col]}]: all satellites",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)
    return save(fig, f"q1b_{col}.png")


def run(data: dict[str, pd.DataFrame]) -> list[Path]:
    """Write one figure per telemetry channel and return the file paths."""
    # path.simplify would drop sub-pixel detail of the long series; keep every sample drawn
    with plt.rc_context({"path.simplify": False}):
        return [plot_channel(data, col) for col in CHANNELS]


if __name__ == "__main__":
    start = time.perf_counter()
    setup_style()
    paths = run(load_all())
    for path in paths:
        print(f"wrote {path}")
    elapsed = time.perf_counter() - start
    print(f"q1b done in {elapsed:.1f} s")
