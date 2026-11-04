"""Plot the linear-chirp excitation trajectory used by PACE.

The phase definition is identical to ``scripts/pace/data_collection.py``::

    phase(t) = 2*pi*(f0*t + (f1-f0)/(2*duration)*t**2)

The resulting position command is ``offset + amplitude * sin(phase)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and save a PACE chirp excitation trajectory.")
    parser.add_argument("--min-frequency", type=float, default=0.1, help="Initial frequency in Hz.")
    parser.add_argument("--max-frequency", type=float, default=10.0, help="Final frequency in Hz.")
    parser.add_argument("--duration", type=float, default=20.0, help="Trajectory duration in seconds.")
    parser.add_argument("--sample-rate", type=float, default=400.0, help="Sampling frequency in Hz.")
    parser.add_argument("--amplitude", type=float, default=0.25, help="Position amplitude in radians.")
    parser.add_argument("--offset", type=float, default=0.0, help="Position offset in radians.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pace_chirp_excitation.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def linear_chirp_position(
    time: np.ndarray,
    min_frequency: float,
    max_frequency: float,
    duration: float,
    amplitude: float,
    offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the position command and instantaneous frequency."""
    frequency_rate = (max_frequency - min_frequency) / duration
    phase = 2.0 * np.pi * (min_frequency * time + 0.5 * frequency_rate * time**2)
    position = offset + amplitude * np.sin(phase)
    instantaneous_frequency = min_frequency + frequency_rate * time
    return position, instantaneous_frequency


def main() -> None:
    args = parse_args()

    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.sample_rate <= 0.0:
        raise ValueError("--sample-rate must be positive.")
    if args.min_frequency < 0.0 or args.max_frequency < args.min_frequency:
        raise ValueError("Frequencies must satisfy 0 <= min-frequency <= max-frequency.")

    num_steps = int(round(args.duration * args.sample_rate))
    time = np.arange(num_steps, dtype=np.float64) / args.sample_rate
    position, instantaneous_frequency = linear_chirp_position(
        time,
        args.min_frequency,
        args.max_frequency,
        args.duration,
        args.amplitude,
        args.offset,
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, position, color="tab:blue", linewidth=1.0)
    axes[0].axhline(args.offset, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_title("PACE Linear-Chirp Joint Excitation")
    axes[0].set_ylabel("Desired joint position [rad]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, instantaneous_frequency, color="tab:orange", linewidth=1.5)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Instantaneous frequency [Hz]")
    axes[1].grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved excitation trajectory to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
