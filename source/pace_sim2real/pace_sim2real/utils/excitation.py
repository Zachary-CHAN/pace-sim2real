# Copyright (c) 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""Excitation-trajectory helpers shared by full-body and single-joint collection."""

from __future__ import annotations

import torch


def get_excitation_parameters(sim2real_cfg, num_joints: int, device: str | torch.device):
    """Return the center, amplitude, and phase offset for every identification joint."""
    if hasattr(sim2real_cfg, "trajectory_center") and hasattr(sim2real_cfg, "trajectory_amplitude"):
        center = sim2real_cfg.trajectory_center.to(device=device)
        amplitude = sim2real_cfg.trajectory_amplitude.to(device=device)
        phase_offset = getattr(
            sim2real_cfg,
            "trajectory_phase_offset",
            torch.zeros_like(center),
        ).to(device=device)
    else:
        # Original ANYmal transformation:
        # (sin(phase) + bias) * direction * scale
        if num_joints != 12:
            raise ValueError(
                "The legacy ANYmal excitation requires exactly 12 identification joints, "
                f"but the task defines {num_joints}."
            )
        direction = torch.tensor(
            [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            device=device,
        )
        bias = torch.tensor([0.0, 0.4, 0.8] * 4, device=device)
        scale = torch.tensor([0.25, 0.5, -2.0] * 4, device=device)
        amplitude = direction * scale
        center = bias * amplitude
        phase_offset = torch.zeros_like(center)

    expected_shape = (num_joints,)
    for name, value in (
        ("trajectory_center", center),
        ("trajectory_amplitude", amplitude),
        ("trajectory_phase_offset", phase_offset),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, but got {tuple(value.shape)}.")

    return center, amplitude, phase_offset


def generate_excitation_trajectory(
    *,
    duration: float,
    sample_rate: float,
    min_frequency: float,
    max_frequency: float,
    center: torch.Tensor,
    amplitude: torch.Tensor,
    phase_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the linear-chirp position trajectory used for PACE identification."""
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, but got {duration}.")
    if sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be positive, but got {sample_rate}.")
    if min_frequency < 0.0 or max_frequency < 0.0:
        raise ValueError("Chirp frequencies must be non-negative.")
    if max_frequency < min_frequency:
        raise ValueError(f"max_frequency ({max_frequency}) must be no smaller than min_frequency ({min_frequency}).")
    if center.ndim != 1 or amplitude.shape != center.shape or phase_offset.shape != center.shape:
        raise ValueError("center, amplitude, and phase_offset must be one-dimensional tensors of equal shape.")

    num_steps = int(duration * sample_rate)
    if num_steps < 1:
        raise ValueError(f"duration * sample_rate must produce at least one sample, but got {duration * sample_rate}.")

    time = torch.arange(num_steps, device=center.device, dtype=center.dtype) / sample_rate
    # Linear chirp: phase = 2*pi*(f0*t + (f1-f0)/(2*duration)*t^2)
    phase = 2.0 * torch.pi * (min_frequency * time + ((max_frequency - min_frequency) / (2.0 * duration)) * time**2)
    trajectory = center.unsqueeze(0) + amplitude.unsqueeze(0) * torch.sin(
        phase.unsqueeze(1) + phase_offset.unsqueeze(0)
    )
    return time, trajectory


def validate_initial_position(
    trajectory: torch.Tensor,
    default_position: torch.Tensor,
    *,
    tolerance: float = 1.0e-4,
) -> None:
    """Ensure an excitation starts at the configured default joint posture."""
    if trajectory.ndim != 2 or default_position.shape != trajectory.shape[1:]:
        raise ValueError(
            "trajectory must have shape (time, joints), and default_position must contain one value per joint."
        )
    initial_error = torch.max(torch.abs(trajectory[0] - default_position))
    if initial_error > tolerance:
        raise ValueError(
            "The configured excitation does not start at the robot default position: "
            f"maximum error is {initial_error.item():.6f} rad."
        )
