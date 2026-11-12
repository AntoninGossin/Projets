"""
SpaceOps take-home test: runs every question and prints the total run time.

Usage: python src/main.py
"""
import time

START = time.perf_counter()

import q1a_per_satellite
import q1b_per_channel
import q1c_beta_angle
import q2_percentages
import q3a_delta_scenarios
import q3b_delta
import q3b_power
import q4_detectors
from telemetry import load_all, load_beta, setup_style, telemetry_window  # noqa: E402


def main() -> None:
    setup_style()

    print("Loading telemetry")
    data = load_all()
    beta = load_beta()

    print("Q1: plots")
    paths = q1a_per_satellite.run(data)
    paths += q1b_per_channel.run(data)
    paths += q1c_beta_angle.run(beta, telemetry_window(data))
    print(f"  {len(paths)} plots written to plots/")

    print("\nQ2: share of the available telemetry points")
    q2_percentages.run(data)

    print("\nQ3a: delta scenarios anchored on bravo")
    paths += q3a_delta_scenarios.run(beta)
    print("\nQ3b: anomalies")
    paths += q3b_delta.run(data)
    paths += q3b_power.run(data)
    print(f"  {len(paths)} plots written to plots/")

    print("\nQ4: detection functions applied to every satellite")
    q4_detectors.run(data)


if __name__ == "__main__":
    main()
    print(f"Total run time: {time.perf_counter() - START:.1f} s")
