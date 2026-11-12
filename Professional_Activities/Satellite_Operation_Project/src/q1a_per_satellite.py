"""
Question 1(a): one figure per satellite with all its telemetry channels.

Each figure has four stacked panels that share the time axis:
  1. gyro rates X, Y, Z with the +/-2 deg/s limit
  2. reaction wheel speeds S, X, Y, Z with the 8000 rpm saturation
  3. bus voltage, low-voltage periods shaded
  4. low-voltage flag (0/1)
The y-limits and the x-limits are the same for all satellites, so the 7 figures
can be compared side by side. Nothing is smoothed or decimated: every raw sample
is plotted, so each peak above 2 deg/s or 8000 rpm stays visible.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from telemetry import (FLAG_COLOR, GYRO_RATE_LIMIT_DEG_S, GYROS, LABELS, LIMIT_COLOR,
                       MAX_GAP, MUTED_COLOR, TEXT_COLOR, UNITS, WHEEL_SATURATION_RPM,
                       WHEELS, axis_color, break_gaps, break_step_gaps, channel,
                       format_time_axis, load_all, save, setup_style, telemetry_window)

FIG_SIZE = (14, 11)
HEIGHT_RATIOS = [3, 3, 2, 1]
LINE_WIDTH = 0.5

# Same y-limits on every figure. The wheel axis starts a bit below 0 so that
# a wheel stuck at 0 rpm (delta) is drawn in full and not cut by the axis line.
GYRO_YLIM = (-5.0, 5.0)
WHEEL_YLIM = (-250.0, 8800.0)
VOLTAGE_YLIM = (6.5, 8.5)


def low_voltage_periods(flag: pd.Series) -> list:
    """(start, end) of each low-voltage period.

    The flag is sampled irregularly. A period starts at a sample equal to 1 and
    ends at the next sample equal to 0 (or at the last 1 if the data stops, or
    at the last 1 before an outage, since the state during an outage is unknown).
    """
    times, values = flag.index, flag.to_numpy()
    gap_before = np.r_[False, np.diff(times.values) > MAX_GAP.to_timedelta64()]
    # a new run starts when the value changes or after an outage
    run_starts = np.flatnonzero(np.r_[True, (values[1:] != values[:-1]) | gap_before[1:]])
    run_ends = np.r_[run_starts[1:], len(values)]  # exclusive

    periods = []
    for first, stop in zip(run_starts, run_ends):
        if values[first] != 1:
            continue
        next_is_zero = stop < len(values) and not gap_before[stop]
        last = stop if next_is_zero else stop - 1
        periods.append((times[first], times[last]))
    return periods


def plot_lines(ax: plt.Axes, df: pd.DataFrame, cols: list) -> None:
    """Draw one line per channel from its sampled values (outages left as breaks)."""
    for col in cols:
        values = channel(df, col)
        if len(values):
            ax.plot(break_gaps(values), color=axis_color(col), lw=LINE_WIDTH, label=LABELS[col])


def add_legend(ax: plt.Axes) -> None:
    """Legend above the panel, right aligned, only when there are 2 series or more."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        legend = ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                           ncol=len(handles), borderaxespad=0.2, handlelength=1.8)
        for line in legend.get_lines():  # the plotted lines are too thin to read in a legend
            line.set_linewidth(1.8)


def add_threshold(ax: plt.Axes, level: float, text: str) -> None:
    """Dashed limit line with its label in the right margin, clear of the data."""
    # zorder: above the grid (1.5) so the grid does not cover it, below the data (2)
    ax.axhline(level, color=LIMIT_COLOR, ls="--", lw=0.8, zorder=1.8)
    margin_text(ax, level, text, va="center")


def margin_text(ax: plt.Axes, y: float, text: str, va: str) -> None:
    """Small note just right of the panel, at data height y."""
    ax.text(1.008, y, text, transform=ax.get_yaxis_transform(), ha="left", va=va,
            fontsize=8, color=MUTED_COLOR, clip_on=False)


def mark_missing(ax: plt.Axes) -> None:
    """Placeholder for a channel the satellite does not telemeter."""
    ax.text(0.5, 0.5, "not telemetered", transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=MUTED_COLOR)


def plot_satellite(name: str, df: pd.DataFrame, window: tuple) -> plt.Figure:
    """Build the four-panel figure of one satellite."""
    fig, (ax_gyro, ax_wheel, ax_volt, ax_flag) = plt.subplots(
        4, 1, figsize=FIG_SIZE, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": 0.28})
    fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.08)

    # 1. gyro rates
    plot_lines(ax_gyro, df, GYROS)
    add_threshold(ax_gyro, GYRO_RATE_LIMIT_DEG_S, f"limit +{GYRO_RATE_LIMIT_DEG_S:g} deg/s")
    add_threshold(ax_gyro, -GYRO_RATE_LIMIT_DEG_S, f"limit \u2212{GYRO_RATE_LIMIT_DEG_S:g} deg/s")
    ax_gyro.set_ylim(*GYRO_YLIM)
    ax_gyro.set_ylabel(f"Gyro rate [{UNITS['gyro_x']}]")
    add_legend(ax_gyro)

    # 2. wheel speeds
    plot_lines(ax_wheel, df, WHEELS)
    add_threshold(ax_wheel, WHEEL_SATURATION_RPM, f"saturation {WHEEL_SATURATION_RPM:.0f} rpm")
    # wheels that never leave 0 rpm lie on the bottom line and hide each other: say so
    stopped = [c[-1].upper() for c in WHEELS if len(channel(df, c)) and channel(df, c).max() == 0]
    if stopped:
        margin_text(ax_wheel, 0, f"Wheel {', '.join(stopped)}: 0 rpm\nwhole window", va="bottom")
    ax_wheel.set_ylim(*WHEEL_YLIM)
    ax_wheel.set_ylabel(f"Wheel speed [{UNITS['wheel_s']}]")
    add_legend(ax_wheel)

    # 3. bus voltage with the low-voltage periods shaded
    ax_volt.set_ylim(*VOLTAGE_YLIM)
    ax_volt.set_ylabel(f"Bus voltage [{UNITS['bus_voltage']}]")
    volts = channel(df, "bus_voltage")
    flag = channel(df, "low_voltage")
    if len(volts):
        ax_volt.plot(break_gaps(volts), color=TEXT_COLOR, lw=LINE_WIDTH,
                     label=LABELS["bus_voltage"])
        periods = low_voltage_periods(flag) if len(flag) else []
        for i, (start, end) in enumerate(periods):
            # only the first span carries the legend entry
            ax_volt.axvspan(start, end, facecolor=FLAG_COLOR, edgecolor=FLAG_COLOR, alpha=0.25,
                            lw=0.6, zorder=0, label="low_voltage = 1" if i == 0 else None)
        add_legend(ax_volt)
    else:
        mark_missing(ax_volt)

    # 4. low-voltage flag as a step line
    if len(flag):
        ax_flag.plot(break_step_gaps(flag), color=TEXT_COLOR, lw=LINE_WIDTH + 0.2,
                     drawstyle="steps-post", label=LABELS["low_voltage"])
    else:
        mark_missing(ax_flag)
    ax_flag.set_ylim(-0.15, 1.15)
    ax_flag.set_yticks([0, 1])
    ax_flag.set_ylabel(f"Low-volt. flag [{UNITS['low_voltage']}]")

    # common x axis
    ax_flag.set_xlim(*window)
    format_time_axis(ax_flag)
    ax_flag.set_xlabel("Time (UTC)")

    # title and subtitle (window of the data)
    fig.text(0.07, 0.975, f"{name}: all telemetry channels", fontsize=15, fontweight="bold",
             ha="left", va="top", color=TEXT_COLOR)
    start, end = window[0].round("min"), window[1].round("min")
    fig.text(0.07, 0.945, f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC",
             fontsize=10, ha="left", va="top", color=MUTED_COLOR)
    return fig


def run(data: dict) -> list:
    """Save one figure per satellite (q1a_<sat>.png) and return the file paths."""
    window = telemetry_window(data)
    paths = []
    # draw every raw segment: no renderer-side path simplification either
    with plt.rc_context({"path.simplify": False}):
        for name, df in data.items():
            fig = plot_satellite(name, df, window)
            paths.append(save(fig, f"q1a_{name}.png"))
    return paths


if __name__ == "__main__":
    t0 = time.perf_counter()
    setup_style()
    paths = run(load_all())
    elapsed = time.perf_counter() - t0
    for path in paths:
        print(f"wrote {path}")
    print(f"q1a done in {elapsed:.1f} s")
