# Copyright (c) 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""Collect a PACE excitation trajectory while moving only one joint."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect PACE excitation data for one joint.")
parser.add_argument(
    "--joint",
    type=str,
    default=None,
    help="Exact name of the joint to excite (required).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-R1-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum chirp frequency in Hz.")
parser.add_argument("--max_frequency", type=float, default=10.0, help="Maximum chirp frequency in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Trajectory duration in seconds.")
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output .pt file. Defaults to data/<robot>/single_joint/<joint>/chirp_data.pt.",
)
parser.add_argument(
    "--no_plot",
    action="store_true",
    help="Do not display the measured and target trajectories.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.joint is None:
    parser.error("--joint is required.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim has started."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import pace_sim2real.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg
from pace_sim2real.utils import project_root
from pace_sim2real.utils.excitation import (
    generate_excitation_trajectory,
    get_excitation_parameters,
    validate_initial_position,
)


def _joint_indices(actuator, num_joints: int, device: torch.device) -> torch.Tensor:
    """Return global articulation indices for an actuator group."""
    indices = actuator.joint_indices
    if isinstance(indices, slice):
        indices = torch.arange(num_joints, device=device)[indices]
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _lock_other_joints(articulation, moving_joint_id: int, env_ids: torch.Tensor) -> None:
    """Lock every non-excited joint at its configured default posture."""
    all_joint_ids = torch.arange(articulation.num_joints, device=articulation.device)
    locked_joint_ids = all_joint_ids[all_joint_ids != moving_joint_id]
    locked_positions = articulation.data.default_joint_pos[:, locked_joint_ids].clone()
    locked_limits = torch.stack((locked_positions, locked_positions), dim=-1)
    articulation.write_joint_position_limit_to_sim(
        locked_limits,
        joint_ids=locked_joint_ids,
        env_ids=env_ids,
    )
    articulation.write_joint_position_to_sim(locked_positions, joint_ids=locked_joint_ids, env_ids=env_ids)
    articulation.write_joint_velocity_to_sim(
        torch.zeros_like(locked_positions),
        joint_ids=locked_joint_ids,
        env_ids=env_ids,
    )


def _configure_synthetic_joint(articulation, joint_id: int, env_ids: torch.Tensor) -> torch.Tensor:
    """Apply the same synthetic parameters used by full-body data collection."""
    device = articulation.device
    armature = torch.full((len(env_ids), 1), 0.01, device=device)
    damping = torch.full((len(env_ids), 1), 4.5, device=device)
    friction = torch.full((len(env_ids), 1), 0.05, device=device)
    encoder_bias = torch.full((len(env_ids), 1), 0.05, device=device)

    articulation.write_joint_armature_to_sim(armature, joint_ids=[joint_id], env_ids=env_ids)
    articulation.data.default_joint_armature[:, joint_id : joint_id + 1] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=[joint_id], env_ids=env_ids)
    articulation.data.default_joint_viscous_friction_coeff[:, joint_id : joint_id + 1] = damping
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=[joint_id], env_ids=env_ids)
    articulation.data.default_joint_friction_coeff[:, joint_id : joint_id + 1] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=[joint_id], env_ids=env_ids)
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_id : joint_id + 1] = friction

    # A PaceDCMotor stores bias per actuator group. Update only the selected
    # column so a single joint can be excited without requiring a split group.
    for actuator in articulation.actuators.values():
        group_joint_ids = _joint_indices(actuator, articulation.num_joints, device)
        local_matches = torch.nonzero(group_joint_ids == joint_id, as_tuple=False).flatten()
        if local_matches.numel() == 0:
            continue
        local_joint_id = local_matches.item()
        group_bias = actuator.encoder_bias.clone()
        group_bias[:, local_joint_id] = encoder_bias[:, 0]
        actuator.update_encoder_bias(group_bias)
        actuator.update_time_lags(torch.full((len(env_ids),), 5, dtype=torch.int, device=device), env_ids)
        actuator.reset(env_ids)
        break
    else:
        raise RuntimeError(f"No actuator group contains joint index {joint_id}.")

    return encoder_bias


def _validate_joint_range(
    articulation,
    joint_id: int,
    center: torch.Tensor,
    amplitude: torch.Tensor,
) -> tuple[float, float]:
    """Check the configured excitation interval against the articulation hard limits."""
    excitation_lower = (center - torch.abs(amplitude)).item()
    excitation_upper = (center + torch.abs(amplitude)).item()
    hard_lower, hard_upper = articulation.data.joint_pos_limits[0, joint_id].tolist()
    tolerance = 1.0e-6
    if excitation_lower < hard_lower - tolerance or excitation_upper > hard_upper + tolerance:
        raise ValueError(
            f"Configured excitation [{excitation_lower:.6f}, {excitation_upper:.6f}] rad exceeds "
            f"the hard joint limits [{hard_lower:.6f}, {hard_upper:.6f}] rad for '{args_cli.joint}'."
        )
    return excitation_lower, excitation_upper


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    articulation = env.unwrapped.scene["robot"]

    identification_joint_order = list(env_cfg.sim2real.joint_order)
    if args_cli.joint not in identification_joint_order:
        available = ", ".join(identification_joint_order)
        env.close()
        raise ValueError(
            f"Joint '{args_cli.joint}' is not in this task's identification joint_order. Available joints: {available}"
        )
    if args_cli.joint not in articulation.joint_names:
        env.close()
        raise ValueError(f"Joint '{args_cli.joint}' does not exist in the loaded articulation.")

    selected_order_index = identification_joint_order.index(args_cli.joint)
    joint_id = articulation.joint_names.index(args_cli.joint)
    joint_ids = torch.tensor([joint_id], dtype=torch.long, device=env.unwrapped.device)

    env.reset()
    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    default_position = articulation.data.default_joint_pos[0, joint_ids].clone()

    all_centers, all_amplitudes, all_phase_offsets = get_excitation_parameters(
        env_cfg.sim2real,
        len(identification_joint_order),
        env.unwrapped.device,
    )
    center = all_centers[selected_order_index : selected_order_index + 1]
    amplitude = all_amplitudes[selected_order_index : selected_order_index + 1]
    phase_offset = all_phase_offsets[selected_order_index : selected_order_index + 1]
    excitation_lower, excitation_upper = _validate_joint_range(articulation, joint_id, center, amplitude)

    sample_rate = 1.0 / env.unwrapped.sim.get_physics_dt()
    time_data, trajectory = generate_excitation_trajectory(
        duration=args_cli.duration,
        sample_rate=sample_rate,
        min_frequency=args_cli.min_frequency,
        max_frequency=args_cli.max_frequency,
        center=center,
        amplitude=amplitude,
        phase_offset=phase_offset,
    )
    validate_initial_position(trajectory, default_position)

    _lock_other_joints(articulation, joint_id, env_ids)
    encoder_bias = _configure_synthetic_joint(articulation, joint_id, env_ids)
    articulation.write_joint_position_to_sim(
        trajectory[0].unsqueeze(0).repeat(env.unwrapped.num_envs, 1) + encoder_bias,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
    articulation.write_joint_velocity_to_sim(
        torch.zeros((env.unwrapped.num_envs, 1), device=env.unwrapped.device),
        joint_ids=joint_ids,
        env_ids=env_ids,
    )

    print(f"[INFO]: Exciting only joint: {args_cli.joint} (articulation index {joint_id})")
    print(f"[INFO]: Default position: {default_position.item():.6f} rad")
    print(f"[INFO]: Excitation range: [{excitation_lower:.6f}, {excitation_upper:.6f}] rad")
    print(f"[INFO]: All other {articulation.num_joints - 1} joints are locked at their default positions.")

    num_steps = time_data.shape[0]
    dof_pos_buffer = torch.zeros((num_steps, 1), device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros((num_steps, 1), device=env.unwrapped.device)
    counter = 0
    while simulation_app.is_running() and counter < num_steps:
        with torch.inference_mode():
            dof_pos_buffer[counter] = articulation.data.joint_pos[0, joint_ids] - encoder_bias[0]
            actions = articulation.data.default_joint_pos.clone()
            actions[:, joint_id] = trajectory[counter, 0]
            env.step(actions)
            dof_target_pos_buffer[counter] = articulation.data.joint_pos_target[0, joint_ids]
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter / sample_rate:.1f} / {args_cli.duration:.1f} seconds")

    control_dt = env.unwrapped.sim.get_physics_dt()
    env.close()
    if counter != num_steps:
        raise RuntimeError(f"Simulation stopped after {counter} of {num_steps} requested samples.")

    output_path = args_cli.output
    if output_path is None:
        output_path = (
            project_root() / "data" / env_cfg.sim2real.robot_name / "single_joint" / args_cli.joint / "chirp_data.pt"
        )
    elif not output_path.is_absolute():
        output_path = project_root() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time": time_data.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "des_dof_pos": dof_target_pos_buffer.cpu(),
            "joint_order": [args_cli.joint],
            "control_dt": control_dt,
            "default_joint_pos": default_position.cpu(),
            "excitation_range": torch.tensor([excitation_lower, excitation_upper]),
        },
        output_path,
    )
    print(f"[INFO]: Saved {num_steps} samples to: {output_path}")

    if not args_cli.no_plot:
        import matplotlib.pyplot as plt

        plt.plot(
            time_data.cpu().numpy(),
            dof_pos_buffer[:, 0].cpu().numpy(),
            label="measured",
        )
        plt.plot(
            time_data.cpu().numpy(),
            dof_target_pos_buffer[:, 0].cpu().numpy(),
            label="target",
            linestyle="dashed",
        )
        plt.title(f"Single-joint excitation: {args_cli.joint}")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
