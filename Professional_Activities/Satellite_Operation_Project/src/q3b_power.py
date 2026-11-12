"""
Question 3b: the power anomaly, satellites compared.

The battery charges in sunlight and drains in eclipse once per orbit (~94 min), so a healthy
satellite is fully charged every orbit: the peak voltage of each ~95 min window stays at
about 8.3 V. A satellite that can no longer recharge shows a falling peak and a low-voltage
flag that stays set longer and longer. Two panels share the time axis: peak bus voltage per
orbit, and share of the orbit with the flag set. Satellites without power telemetry (delta)
are left out and named in a footnote.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

from q1c_beta_angle import group_by_end_value, spread
from q2_percentages import share_of_time
from telemetry import (MAX_GAP, MUTED_COLOR, POWER, SAT_COLORS, SATELLITES, TEXT_COLOR, channel,
                       format_time_axis, load_all, save, setup_style, telemetry_window)

FULL_CHARGE_V = 8.3   # peak of a healthy satellite
MIN_COVERAGE = 0.5    # a window needs telemetry over at least this part of its duration
LABEL_GAP_PT = 12.0   # minimum vertical distance between two end labels, in points
DOT_STEP_PT = 7.0     # spacing of the colored dots in front of a label
# (column, y label, y limits, labels closer than this at the line end are merged)
PANELS = [("peak_v", "Peak bus voltage per orbit [V]", (6.5, 8.5), 0.05),
          ("low_pct", "Share of the orbit with low-voltage flag set [%]", (-5, 105), 5)]


def per_orbit(df: pd.DataFrame, period: str = "95min") -> pd.DataFrame:
    """Peak bus voltage [V] and share of time with low_voltage == 1 [%] per window.

    Windows are consecutive and start on the hour of the first sample, so all satellites
    share the same ones. A window is NaN when telemetry covers less than MIN_COVERAGE of it
    (outage): its peak could miss the sunlit part of the orbit. The last window is dropped
    when the end of the data cuts it short. The index is the window centre.
    """
    origin = df.index[0].floor("h")
    volt = channel(df, "bus_voltage")
    peak = volt.resample(period, origin=origin).max()
    low = (channel(df, "low_voltage") == 1).resample(period, origin=origin).apply(share_of_time)
    # time covered by samples, ignoring the silences longer than MAX_GAP
    step = volt.index.to_series().diff()
    seen = step.where(step <= MAX_GAP, pd.Timedelta(0)).resample(period, origin=origin).sum()
    out = pd.DataFrame({"peak_v": peak, "low_pct": low})
    out = out.where(seen / pd.Timedelta(period) >= MIN_COVERAGE)
    out = out[out.index + pd.Timedelta(period) <= df.index[-1]]
    return out.shift(freq=pd.Timedelta(period) / 2)


def outages(df: pd.DataFrame, min_gap: str = "2h") -> list:
    """(start, end) of every silence of the bus voltage longer than min_gap."""
    t = channel(df, "bus_voltage").index.to_series()
    long = t.diff() > pd.Timedelta(min_gap)
    return list(zip(t.shift()[long], t[long]))


def label_line_ends(ax: plt.Axes, last: pd.Series, tol: float) -> None:
    """Name the lines right of the axes at their last value; lines ending together share a label."""
    fig = ax.figure
    groups = group_by_end_value(last, tol)
    ymin, ymax = ax.get_ylim()
    pt_per_unit = ax.get_window_extent().height * 72 / fig.dpi / (ymax - ymin)
    y_text = spread([last[g].mean() for g in groups], LABEL_GAP_PT / pt_per_unit)
    for names, y in zip(groups, y_text):
        for i, name in enumerate(names):  # one small colored dot per satellite
            at = mtransforms.offset_copy(ax.get_yaxis_transform(), fig=fig, x=8 + i * DOT_STEP_PT,
                                         units="points")
            ax.plot([1], [y], marker="o", markersize=4.5, color=SAT_COLORS[name],
                    linestyle="none", transform=at, clip_on=False)
        at = mtransforms.offset_copy(ax.get_yaxis_transform(), fig=fig,
                                     x=8 + len(names) * DOT_STEP_PT, units="points")
        ax.text(1, y, " / ".join(names), transform=at, ha="left", va="center",
                fontsize=9, color=TEXT_COLOR)  # text stays neutral, never in a series color


def run(data: dict[str, pd.DataFrame]) -> list[Path]:
    """Plot the peak voltage and the low-voltage share per orbit of every satellite with power."""
    orbits = {n: per_orbit(df) for n, df in data.items() if df[POWER].notna().any().any()}
    shown = [n for n in SATELLITES if n in orbits]
    missing = [n for n in SATELLITES if n not in orbits]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.85, top=0.85, bottom=0.11, hspace=0.12)
    fig.text(0.07, 0.95, "echo and bravo no longer recharge their battery", fontsize=15,
             fontweight="bold")
    fig.text(0.07, 0.915, "Per window of 95 min (about one orbit). A healthy satellite is fully "
             "charged every orbit and its low-voltage flag stays off.", fontsize=10, color=MUTED_COLOR)

    # draw the healthiest first, so the anomalous lines end up on top of the overlapping ones
    order = sorted(orbits, key=lambda n: -orbits[n]["peak_v"].mean())
    handles = {}
    for ax, (col, ylabel, ylim, tol) in zip(axes, PANELS):
        for name in order:
            handles[name], = ax.plot(orbits[name][col], color=SAT_COLORS[name], marker="o",
                                     markersize=3, linewidth=1.2)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        label_line_ends(ax, pd.Series({n: o[col].dropna().iloc[-1] for n, o in orbits.items()}), tol)

    # reference line at the healthy full-charge level
    top = axes[0]
    top.axhline(FULL_CHARGE_V, color=MUTED_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    top.text(0.005, FULL_CHARGE_V + 0.035, f"healthy: full charge, about {FULL_CHARGE_V} V",
             transform=top.get_yaxis_transform(), va="bottom", fontsize=8, color=MUTED_COLOR)

    # long silences are shaded in both panels; the lines are broken there (NaN windows)
    for name in orbits:
        for t0, t1 in outages(data[name]):
            for ax in axes:
                ax.axvspan(t0, t1, color=MUTED_COLOR, alpha=0.12, linewidth=0, zorder=0)
            hours = (t1 - t0).total_seconds() / 3600
            top.text(t0 + (t1 - t0) / 2, 0.06, f"{name}: no telemetry for {hours:.0f} h",
                     transform=top.get_xaxis_transform(), ha="center", fontsize=9, color=MUTED_COLOR)

    x0, x1 = telemetry_window(data)
    top.set_xlim(x0, x1)
    format_time_axis(axes[1])
    axes[1].set_xlabel("Time (UTC)")
    top.legend([handles[n] for n in shown], shown, loc="lower left", bbox_to_anchor=(0, 1.0),
               ncol=len(shown), borderaxespad=0.3)
    if missing:
        fig.text(0.07, 0.01, f"{', '.join(missing)} not shown: no power telemetry (bus voltage is "
                 "a constant 0 V fill value). A window with telemetry for less than half of its "
                 "time is left blank.", fontsize=8, color=MUTED_COLOR)
    return [save(fig, "q3b_power.png")]


if __name__ == "__main__":
    start = time.perf_counter()
    setup_style()
    paths = run(load_all())
    elapsed = time.perf_counter() - start
    for path in paths:
        print(f"wrote {path}")
    print(f"q3b power done in {elapsed:.1f} s")
