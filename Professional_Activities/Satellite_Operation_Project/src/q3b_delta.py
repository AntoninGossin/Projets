"""
Question 3b: delta, a satellite that tumbles because its reaction wheels have stopped.

One zoomed figure on 2019-06-01 00:00 -> end of data, two panels sharing the time axis:
  1. reaction wheel speeds with the 8000 rpm saturation (runs >= 5 min shaded)
  2. gyro rates X, Y, Z and the body rate |w| = sqrt(gx^2 + gy^2 + gz^2), with the +/-2 deg/s limit
Two events are found in the data and marked: the last non-zero wheel sample, and the
start of the final stretch where |w| stays above 2 deg/s.
"""
from __future__ import annotations

import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q1a_per_satellite import GYRO_YLIM, WHEEL_YLIM, add_legend, add_threshold, margin_text
from telemetry import (GYRO_RATE_LIMIT_DEG_S, GYROS, LABELS, LIMIT_COLOR, MUTED_COLOR, TEXT_COLOR,
                       WHEEL_SATURATION_RPM, WHEELS, axis_color, break_gaps, channel, load_all,
                       save, setup_style)

WINDOW_START = pd.Timestamp("2019-06-01")
MIN_RUN = pd.Timedelta("5min")  # shorter saturation runs are not shaded
RATE_MEDIAN = "5min"            # rolling median of |w|, trailing window


def saturation_runs(wheels: pd.DataFrame) -> list:
    """(start, end) of each run where at least one wheel is saturated for >= MIN_RUN."""
    saturated = (wheels >= WHEEL_SATURATION_RPM).any(axis=1)
    run_id = saturated.ne(saturated.shift()).cumsum()
    runs = [(seg.index[0], seg.index[-1]) for _, seg in saturated.groupby(run_id) if seg.iloc[0]]
    return [(start, end) for start, end in runs if end - start >= MIN_RUN]


def body_rate(df: pd.DataFrame) -> pd.Series:
    """|w| [deg/s]: norm of the three gyro rates (rows with all 3 axes), 5-min rolling median."""
    gyro = df[GYROS].dropna()
    return np.sqrt((gyro ** 2).sum(axis=1)).rolling(RATE_MEDIAN).median()


def mark_event(axes: list, time_: pd.Timestamp, text: str, height: float) -> None:
    """Thin vertical line through all panels, labelled in the first one (height: axes fraction)."""
    for ax in axes:
        ax.axvline(time_, color=MUTED_COLOR, lw=0.8, zorder=1.8)
    axes[0].annotate(f"{text}\n{time_:%b %d %H:%M} UTC", (mdates.date2num(time_), height),
                     xycoords=axes[0].get_xaxis_transform(), xytext=(4, 0),
                     textcoords="offset points", ha="left", va="top", fontsize=8, color=MUTED_COLOR)


def run(data: dict) -> list:
    """Save q3b_delta_tumbling.png and return its path in a list."""
    df = data["delta"]
    end = df.index[-1]
    wheels = df[WHEELS].dropna(how="all")
    rate = body_rate(df)

    # Events, computed from the data
    t_wheels_off = wheels.index[wheels.max(axis=1) > 0][-1]       # last non-zero wheel sample
    t_tumble = rate.index[rate.index > rate.index[rate <= GYRO_RATE_LIMIT_DEG_S][-1]][0]
    runs = [r for r in saturation_runs(wheels) if r[0] >= WINDOW_START]  # inside the window
    dead = [c[-1].upper() for c in WHEELS if channel(df, c).max() == 0]   # wheels stuck at 0 rpm
    x_zero = (channel(df, "wheel_x") == 0).mean()
    span_h = (end - df.index[0]).total_seconds() / 3600

    win = df.loc[WINDOW_START:]
    # draw every raw segment: no renderer-side path simplification
    with plt.rc_context({"path.simplify": False}):
        fig, (ax_w, ax_g) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                         gridspec_kw={"hspace": 0.25})
        fig.subplots_adjust(left=0.07, right=0.88, top=0.865, bottom=0.09)

        # 1. wheel speeds
        for col in WHEELS:
            ax_w.plot(break_gaps(channel(win, col)), color=axis_color(col), lw=0.5,
                      label=LABELS[col])
        for i, (start, stop) in enumerate(runs):  # only the first span carries the legend entry
            ax_w.axvspan(start, stop, facecolor=LIMIT_COLOR, alpha=0.15, lw=0, zorder=0,
                         label="saturated ≥ 5 min" if i == 0 else None)
        add_threshold(ax_w, WHEEL_SATURATION_RPM, f"saturation {WHEEL_SATURATION_RPM:.0f} rpm")
        margin_text(ax_w, 0, f"Wheel {', '.join(dead)}: 0 rpm\nwhole {span_h:.0f} h", va="bottom")
        margin_text(ax_w, 1500, f"Wheel X: 0 rpm in\n{x_zero:.1%} of samples", va="bottom")
        ax_w.set_ylim(*WHEEL_YLIM)
        ax_w.set_ylabel("Wheel speed [rpm]")
        add_legend(ax_w)

        # 2. gyro rates and body rate
        for col in GYROS:
            ax_g.plot(break_gaps(channel(win, col)), color=axis_color(col), lw=0.4,
                      label=LABELS[col])
        ax_g.plot(rate.loc[WINDOW_START:], color=TEXT_COLOR, lw=1.8,
                  label="|w| body rate (5-min rolling median)")
        add_threshold(ax_g, GYRO_RATE_LIMIT_DEG_S, f"limit +{GYRO_RATE_LIMIT_DEG_S:g} deg/s")
        add_threshold(ax_g, -GYRO_RATE_LIMIT_DEG_S, f"limit −{GYRO_RATE_LIMIT_DEG_S:g} deg/s")
        ax_g.set_ylim(*GYRO_YLIM)
        ax_g.set_ylabel("Rate [deg/s]")
        add_legend(ax_g)

        # events, labelled in the empty part of the wheel panel
        mark_event([ax_w, ax_g], t_wheels_off, "all wheels at 0 rpm", 0.88)
        mark_event([ax_w, ax_g], t_tumble, "|w| > 2 deg/s for good", 0.60)

        # common x axis: hourly ticks over a 34 h window
        ax_g.set_xlim(WINDOW_START, end)
        ax_g.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        ax_g.xaxis.set_minor_locator(mdates.HourLocator())
        ax_g.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax_g.set_xlabel("Time (UTC)")

        fig.text(0.07, 0.975, "delta: two wheels dead, the last wheels stop, the satellite tumbles",
                 fontsize=15, fontweight="bold", ha="left", va="top", color=TEXT_COLOR)
        fig.text(0.07, 0.935, "With no wheel left there is no attitude control: the body rate keeps "
                 "growing, likely fed by the momentum the wheels stored as they spin down.",
                 fontsize=10, ha="left", va="top", color=MUTED_COLOR)
        return [save(fig, "q3b_delta_tumbling.png")]


if __name__ == "__main__":
    t0 = time.perf_counter()
    setup_style()
    paths = run(load_all())
    elapsed = time.perf_counter() - t0
    for path in paths:
        print(f"wrote {path}")
    print(f"q3b delta done in {elapsed:.1f} s")
