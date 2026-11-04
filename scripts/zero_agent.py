# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--video", action="store_true", default=False, help="Record one video and then exit.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video in environment steps.")
parser.add_argument(
    "--video_folder",
    type=str,
    default="videos/zero_agent",
    help="Directory in which the recorded MP4 file is saved.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Camera rendering must be enabled before Isaac Sim is launched.
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import pace_sim2real.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    if args_cli.video_length <= 0:
        raise ValueError("--video_length must be greater than zero.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        # Gymnasium 1.2 stops when recorded_frames > video_length, so passing
        # N directly records N + 1 frames.  The first frame is captured by the
        # first env.step() (step_id == 0); use N - 1 to finalize on frame N.
        record_video_stop_length = max(args_cli.video_length - 1, 0)
        recording_timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        video_name_prefix = f"zero-agent-{recording_timestamp}"
        video_kwargs = {
            "video_folder": args_cli.video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": record_video_stop_length,
            "name_prefix": video_name_prefix,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print(
            f"[INFO]: Recording exactly {args_cli.video_length} frames at "
            f"{env.frames_per_sec} FPS to: "
            f"{args_cli.video_folder}/{video_name_prefix}-step-0.mp4"
        )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # PACE uses absolute joint-position actions (use_default_offset=False).
    # A zero action therefore means "command every joint to 0 rad", not
    # "hold the default pose".  Use the configured pose as the hold target.
    articulation = env.unwrapped.scene["robot"]
    hold_actions = articulation.data.default_joint_pos.clone()
    if hold_actions.shape != env.action_space.shape:
        raise RuntimeError(
            "The zero-agent hold target does not match the action space: "
            f"default_joint_pos={tuple(hold_actions.shape)}, action_space={env.action_space.shape}. "
            "Use a task-specific action-to-joint mapping for partial-joint tasks."
        )

    # Only the joints selected for system identification are allowed to move.
    # In the R1 task, sim2real.joint_order contains the 12 leg joints.  Keep all
    # remaining joints exactly at the initial pose configured in assets/r1.py.
    movable_joint_names = set(env_cfg.sim2real.joint_order)
    fixed_joint_ids = torch.tensor(
        [
            joint_id
            for joint_id, joint_name in enumerate(articulation.joint_names)
            if joint_name not in movable_joint_names
        ],
        dtype=torch.long,
        device=env.unwrapped.device,
    )
    fixed_joint_pos = articulation.data.default_joint_pos[:, fixed_joint_ids].clone()
    fixed_joint_vel = torch.zeros_like(fixed_joint_pos)

    print(
        f"[INFO]: Movable joints ({len(movable_joint_names)}): "
        f"{[name for name in articulation.joint_names if name in movable_joint_names]}"
    )
    print(
        f"[INFO]: Joints held at their configured initial pose ({fixed_joint_ids.numel()}): "
        f"{[articulation.joint_names[index] for index in fixed_joint_ids.tolist()]}"
    )

    # simulate environment
    timestep = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Hold the configured initial pose. Clone because the environment's
            # action manager may retain or modify the action tensor.
            actions = hold_actions.clone()
            # apply actions
            env.step(actions)

            # Position targets alone still permit small transient motion.  Write
            # the non-identified joints back to their initial state after every
            # physics step so that only the 12 R1 leg DoFs can move.
            if fixed_joint_ids.numel() > 0:
                articulation.write_joint_state_to_sim(
                    fixed_joint_pos,
                    fixed_joint_vel,
                    joint_ids=fixed_joint_ids,
                )

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                # For video_length == 1 the wrapper receives zero (unlimited),
                # therefore explicitly finalize that single captured frame.
                if env.recording:
                    env.stop_recording()
                print(f"[INFO]: Video recording completed: {timestep} frames.")
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
