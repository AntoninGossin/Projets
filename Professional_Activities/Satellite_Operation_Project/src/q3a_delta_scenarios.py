"""
Question 3(a): delta beta angle, observed data and reference-anchored scenarios.

Scenario = observed bravo + modelled differential response:
    beta_delta_estimated(t) = beta_bravo_observed(t) + delta_beta_simulated(t)
    delta_beta_simulated(t) = beta_model(bravo orbit + change)(t) - beta_model(bravo orbit)(t)

bravo's orbit (inclination, RAAN, precession rate) is fitted on its beta curve
(q3a_delta_orbit.py). Each scenario changes it by a fixed amount: altitude +/-10 km or
inclination +/-0.04 deg, and the J2 precession rate follows (rate ~ a^-3.5 cos i). All
scenarios start at the delta - bravo offset observed on the first day (+0.248 deg).

Retrospective comparison: bravo data are used throughout, so these are not independent
orbital predictions.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from q3a_delta_orbit import anchored_response, find_date_offset, fit_plane, sun_direction
from telemetry import MUTED_COLOR, TEXT_COLOR, load_beta, save, setup_style

REFERENCE = "bravo"
TITLE_COLOR = "#1f3550"
BRAVO_COLOR = "#2a8c7e"
# label, altitude change [km], inclination change [deg], colour, line style
SCENARIOS = [
    ("No altitude / inclination difference", 0.0, 0.0, "#7d8590", ":"),
    ("Altitude +10 km", 10.0, 0.0, "#2a64ad", "-"),
    ("Altitude -10 km", -10.0, 0.0, "#86acd8", "-"),
    ("Inclination +0.04 deg", 0.0, 0.04, "#c0612b", "--"),
    ("Inclination -0.04 deg", 0.0, -0.04, "#e2a86a", "--"),
]


def scenario_gaps(beta: pd.DataFrame) -> dict[str, pd.Series]:
    """Modelled delta - bravo beta of every scenario, anchored on the first-day offset."""
    offset = find_date_offset(beta[REFERENCE])
    ref = fit_plane(beta[REFERENCE], offset)
    sun = sun_direction(beta.index, offset)
    days = (beta.index - beta.index[0]).days.to_numpy(float)
    gap0 = beta["delta"].iloc[0] - beta[REFERENCE].iloc[0]
    return {label: pd.Series(anchored_response(sun, days, ref, gap0, d_alt, d_incl), index=beta.index)
            for label, d_alt, d_incl, _, _ in SCENARIOS}


def run(beta: pd.DataFrame) -> list[Path]:
    """Two panels: beta angles (top) and delta - bravo (bottom), observed and scenarios."""
    gaps = scenario_gaps(beta)
    bravo, delta = beta[REFERENCE], beta["delta"]
    observed_gap = delta - bravo

    fig, (ax_beta, ax_gap) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 1], "hspace": 0.22})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.07)

    for label, _, _, color, style in SCENARIOS:
        width = 1.6 if style == ":" else 1.8
        ax_beta.plot(beta.index, bravo + gaps[label], color=color, ls=style, lw=width, label=label)
        ax_gap.plot(beta.index, gaps[label], color=color, ls=style, lw=width)
    ax_beta.plot(beta.index, bravo, color=BRAVO_COLOR, lw=1.6, label="Bravo: observed reference")
    ax_beta.plot(beta.index, delta, color=TEXT_COLOR, lw=2.6, label="Delta: observed")
    ax_gap.plot(beta.index, observed_gap, color=TEXT_COLOR, lw=2.6)

    ax_beta.set_ylabel("Beta angle (deg)")
    ax_beta.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95, edgecolor="#d5dbe2",
                   fontsize=9)
    ax_gap.axhline(0, color=MUTED_COLOR, lw=0.8)
    ax_gap.set_ylabel("Delta - Bravo beta angle (deg)")
    final = observed_gap.iloc[-1]
    ax_gap.annotate(f"Observed final offset: {final:.3f}°", xy=(beta.index[-1], final),
                    xytext=(-280, 22), textcoords="offset points", fontsize=9, color=TEXT_COLOR,
                    arrowprops=dict(arrowstyle="-", color=TEXT_COLOR, lw=0.9))
    ax_gap.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, 2)))
    ax_gap.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_gap.set_xlim(beta.index[0] - pd.Timedelta(days=5), beta.index[-1] + pd.Timedelta(days=5))
    ax_gap.set_xlabel("Date (daily samples)")

    fig.text(0.08, 0.965, "Delta beta angle: observed data and reference-anchored scenarios",
             fontsize=15, fontweight="bold", color=TITLE_COLOR)
    fig.text(0.08, 0.905, "Scenario = observed Bravo + modelled differential response. All scenarios "
             f"match the initial observed offset ({observed_gap.iloc[0]:+.3f}°).\nRetrospective "
             "comparison: Bravo data are used throughout. These are not independent orbital predictions.",
             fontsize=9.5, color=MUTED_COLOR, linespacing=1.5)

    print(f"observed delta - bravo: {observed_gap.iloc[0]:+.3f} -> {final:+.3f} deg")
    for label, *_ in SCENARIOS:
        print(f"  {label:<38} ends at {gaps[label].iloc[-1]:+.3f} deg")
    return [save(fig, "q3a_delta_drift.png")]


if __name__ == "__main__":
    start = time.perf_counter()
    setup_style()
    paths = run(load_beta())
    for path in paths:
        print(f"wrote {path}")
    print(f"q3a delta scenarios done in {time.perf_counter() - start:.1f} s")
