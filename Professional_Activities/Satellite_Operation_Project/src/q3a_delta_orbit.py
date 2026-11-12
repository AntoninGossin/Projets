"""
Question 3(a), delta: why does its beta angle drift away from the rest of flock 2 (bravo, echo, golf)?

Physics
-------
The beta angle is the angle between the Sun direction s and the orbital plane:
    beta = asin(s . h),   h = (sin i sin RAAN, -sin i cos RAAN, cos i)   (orbit normal)
The Earth's flattening (J2) makes the plane turn around the polar axis (nodal precession):
    RAAN_rate = -1.5 n J2 (R / a)^2 cos i,   n = sqrt(mu / a^3)   ->   RAAN_rate ~ a^-3.5 cos i
A satellite slightly higher (larger a) or slightly less inclined (cos i closer to 0) turns
slower, and the gap in RAAN, hence in beta, grows with time.

Method
------
1. Fit this model (inclination, initial RAAN, RAAN rate and a slow change of that rate) to
   the beta curves. The fit only works once the beta file is moved back by about 32 days:
   its dates are offset from the Sun position the values were computed with. With that
   offset, every curve is reproduced to ~0.005 deg.
2. Scenario for delta = observed bravo + modelled differential response, i.e. the beta
   difference between bravo's fitted orbit and the same orbit with one change: none,
   altitude only, or inclination only. Every scenario starts from the gap observed on the
   first day (+0.248 deg), so the size of the change is its only free parameter. The best
   scenario tells which difference the shape of the drift supports.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from telemetry import MUTED_COLOR, TEXT_COLOR, load_beta, save, setup_style

FLOCK_2_STEADY = ["bravo", "echo", "golf"]  # flock 2 without delta
REFERENCE = "bravo"  # flock-2 satellite the delta scenarios are built on
EARTH_RADIUS_KM = 6378.137
REFERENCE_ALTITUDE_KM = 490.0  # flock 2, from the orbital period (q3a_flocks.py)
SCENARIO_COLORS = {"same orbit": MUTED_COLOR, "altitude only": "#2a78d6",
                   "inclination only": "#eb6834"}


def sun_direction(dates: pd.DatetimeIndex, offset_days: float = 0.0) -> np.ndarray:
    """Unit vector to the Sun in equatorial coordinates, 3 x N (low-precision almanac, ~0.01 deg)."""
    n = dates.to_julian_date().to_numpy() - 2451545.0 + offset_days  # days since J2000
    mean_longitude = 280.460 + 0.9856474 * n
    anomaly = np.radians(357.528 + 0.9856003 * n)
    ecl_lon = np.radians(mean_longitude + 1.915 * np.sin(anomaly) + 0.020 * np.sin(2 * anomaly))
    obliquity = np.radians(23.439 - 4e-7 * n)
    return np.stack([np.cos(ecl_lon), np.cos(obliquity) * np.sin(ecl_lon),
                     np.sin(obliquity) * np.sin(ecl_lon)])


def beta_model(sun: np.ndarray, days: np.ndarray, incl: float, raan0: float, rate: float,
               rate_change: float = 0.0) -> np.ndarray:
    """Beta angle [deg] of a circular orbit whose plane precesses at rate (+ rate_change * t)."""
    raan = np.radians(raan0 + rate * days + 0.5 * rate_change * days ** 2)
    i = np.radians(incl)
    normal = np.stack([np.sin(i) * np.sin(raan), -np.sin(i) * np.cos(raan),
                       np.full_like(raan, np.cos(i))])
    return np.degrees(np.arcsin((sun * normal).sum(axis=0)))


def fit_plane(curve: pd.Series, offset_days: float) -> dict:
    """Fit inclination, initial RAAN, RAAN rate and its change to one beta curve."""
    sun = sun_direction(curve.index, offset_days)
    days = (curve.index - curve.index[0]).days.to_numpy(float)
    y = curve.to_numpy()
    best = None
    for raan0 in range(0, 360, 30):  # the RAAN is unknown: try several starting points
        fit = least_squares(lambda p: beta_model(sun, days, *p) - y, [97.4, raan0, 0.99, 0.0],
                            bounds=([90, -720, 0.8, -1e-3], [105, 720, 1.2, 1e-3]), x_scale="jac")
        if best is None or fit.cost < best.cost:
            best = fit
    incl, raan0, rate, rate_change = best.x
    return dict(incl=incl, raan0=raan0, rate=rate, rate_change=rate_change,
                rms=float(np.sqrt(np.mean(best.fun ** 2))))


def find_date_offset(curve: pd.Series) -> float:
    """Offset [days] between the beta file dates and the Sun position, from one curve."""
    scan = {offset: fit_plane(curve, offset)["rms"] for offset in range(-60, 61, 4)}
    start = min(scan, key=scan.get)
    fine = {o: fit_plane(curve, o)["rms"] for o in np.arange(start - 4, start + 4.01, 0.25)}
    return float(min(fine, key=fine.get))


def anchored_response(sun: np.ndarray, days: np.ndarray, ref: dict, gap0: float,
                      d_alt_km: float = 0.0, d_incl: float = 0.0) -> np.ndarray:
    """Modelled differential response: beta of the reference orbit changed by d_alt_km and
    d_incl, minus beta of the reference orbit [deg].

    J2: relative change of the precession rate = -3.5 da/a - tan(i) di. The initial RAAN
    offset is chosen so that the response starts at gap0, the gap observed on the first day.
    """
    ref_beta = beta_model(sun, days, ref["incl"], ref["raan0"], ref["rate"], ref["rate_change"])
    a_ref = EARTH_RADIUS_KM + REFERENCE_ALTITUDE_KM
    factor = 1 - 3.5 * d_alt_km / a_ref - np.tan(np.radians(ref["incl"])) * np.radians(d_incl)

    def response(d_raan0: float) -> np.ndarray:
        changed = beta_model(sun, days, ref["incl"] + d_incl, ref["raan0"] + d_raan0,
                             ref["rate"] * factor, ref["rate_change"])
        return changed - ref_beta

    d_raan0 = brentq(lambda x: response(x)[0] - gap0, -5, 5)
    return response(d_raan0)


def delta_scenarios(beta: pd.DataFrame, offset_days: float) -> tuple[dict, dict]:
    """Scenario for delta = observed bravo + modelled differential response.

    The response is the beta difference between bravo's fitted orbit and the same orbit with
    one change. Its initial RAAN offset is set so that it matches the delta - bravo gap
    observed on the first day; the size of the change is the only free parameter.
    Returns (bravo's fitted plane, scenarios).
    """
    ref = fit_plane(beta[REFERENCE], offset_days)
    sun = sun_direction(beta.index, offset_days)
    days = (beta.index - beta.index[0]).days.to_numpy(float)
    observed_gap = (beta["delta"] - beta[REFERENCE]).to_numpy()

    def anchored(**change) -> np.ndarray:
        return anchored_response(sun, days, ref, observed_gap[0], **change)

    # name: (response as a function of the size of the change, starting guess)
    shapes = {"same orbit": (lambda size: anchored(), None),
              "altitude only": (lambda size: anchored(d_alt_km=size), 5.0),
              "inclination only": (lambda size: anchored(d_incl=size), -0.02)}
    scenarios = {}
    for name, (shape, guess) in shapes.items():
        size = 0.0
        if guess is not None:
            size = least_squares(lambda p: shape(p[0]) - observed_gap, [guess]).x[0]
        gap = shape(size)
        scenarios[name] = dict(size=size, gap=gap,
                               rms=float(np.sqrt(np.mean((observed_gap - gap) ** 2))))
    return ref, scenarios


def plot_drift(beta: pd.DataFrame, ref: dict, scenarios: dict) -> Path:
    """Observed delta - bravo beta against the scenarios, and what each scenario leaves."""
    observed = beta["delta"] - beta[REFERENCE]
    fig, (ax_gap, ax_res) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 2], "hspace": 0.32})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.08)

    ax_gap.plot(observed.index, observed, color=TEXT_COLOR, lw=2.4, label="observed")
    for name, sc in scenarios.items():
        label = name if name == "same orbit" else f"{name}: {scenario_size(name, sc)}"
        ax_gap.plot(beta.index, sc["gap"], color=SCENARIO_COLORS[name],
                    lw=1.3, ls=":" if name == "same orbit" else "--", label=label)
        if name != "same orbit":
            ax_res.plot(beta.index, observed - sc["gap"], color=SCENARIO_COLORS[name],
                        lw=1.1, label=f"{name}: rms {sc['rms']:.4f} deg")
    ax_gap.set_ylabel(f"delta minus {REFERENCE} [deg]")
    ax_gap.legend(loc="upper left")
    ax_res.axhspan(-ref["rms"], ref["rms"], color=MUTED_COLOR, alpha=0.15, lw=0,
                   label=f"model noise ({REFERENCE} fit, rms {ref['rms']:.4f} deg)")
    ax_res.axhline(0, color=MUTED_COLOR, lw=0.8)
    ax_res.set_ylabel("observed minus scenario [deg]")
    ax_res.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)  # above the panel, clear of the data
    ax_res.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, 2)))
    ax_res.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_res.set_xlim(beta.index[0], beta.index[-1])
    ax_res.set_xlabel("Date (beta file)")

    fig.text(0.08, 0.965, "delta beta angle relative to bravo: observed data and modelled scenarios",
             fontsize=12, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.08, 0.925, f"Scenario = observed {REFERENCE.capitalize()} + modelled differential "
             f"response. All scenarios match the initial observed offset ({observed.iloc[0]:+.3f}°).",
             fontsize=9, color=MUTED_COLOR)
    return save(fig, "q3a_delta_fit.png")


def scenario_size(name: str, scenario: dict) -> str:
    """Human-readable size of the fitted change."""
    if name == "altitude only":
        return f"{scenario['size']:+.1f} km"
    if name == "inclination only":
        return f"{scenario['size']:+.3f} deg"
    return "-"


def run(beta: pd.DataFrame) -> list[Path]:
    """Find the date offset, fit every orbital plane, test the delta scenarios, plot."""
    offset = find_date_offset(beta[REFERENCE])
    print(f"beta file dates offset from the Sun position: {offset:+.2f} days")

    print(f"\n{'satellite':<10}{'incl deg':>10}{'RAAN rate deg/day':>19}{'fit rms deg':>13}")
    planes = {}
    for name in beta.columns:
        planes[name] = fit_plane(beta[name], offset)
        p = planes[name]
        print(f"{name:<10}{p['incl']:>10.3f}{p['rate']:>19.5f}{p['rms']:>13.4f}")

    f2 = {key: np.mean([planes[s][key] for s in FLOCK_2_STEADY]) for key in ("incl", "rate", "raan0")}
    d_rate = planes["delta"]["rate"] - f2["rate"]
    print(f"\ndelta vs bravo, echo, golf: plane precesses {d_rate * 365.25:+.2f} deg/yr, "
          f"fitted inclination {planes['delta']['incl'] - f2['incl']:+.3f} deg")
    # if the rate difference was constant, when were the two planes aligned?
    d_raan0 = (planes["delta"]["raan0"] - f2["raan0"] + 180) % 360 - 180
    aligned = beta.index[0] - pd.Timedelta(days=float(d_raan0 / d_rate))
    print(f"planes aligned around {aligned:%Y-%m-%d} (beta file dates), extrapolating a constant drift")

    ref, scenarios = delta_scenarios(beta, offset)
    gap0 = beta["delta"].iloc[0] - beta[REFERENCE].iloc[0]
    print(f"\nscenarios = observed {REFERENCE} + modelled response, all starting at the observed gap {gap0:+.3f} deg")
    for name, sc in scenarios.items():
        print(f"  {name:<17} change {scenario_size(name, sc):>11}   rms {sc['rms']:.4f} deg")
    return [plot_drift(beta, ref, scenarios)]


if __name__ == "__main__":
    start = time.perf_counter()
    setup_style()
    paths = run(load_beta())
    for path in paths:
        print(f"wrote {path}")
    print(f"q3a delta orbit done in {time.perf_counter() - start:.1f} s")
