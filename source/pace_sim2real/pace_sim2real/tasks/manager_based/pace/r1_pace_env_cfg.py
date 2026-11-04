# Copyright (c) 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""PACE system-identification environment for the R1 humanoid."""

import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.assets import R1_CFG
from pace_sim2real.utils import PaceDCMotorCfg

# Keep this order identical to the column order in the real-robot data file.
R1_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


R1_TRUNK_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=["waist_.*_joint"],
    saturation_effort=300.0,
    effort_limit=300.0,
    velocity_limit=10.0,
    stiffness={".*": 15000.0},
    damping={".*": 300.0},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=10,
)

R1_LEGS_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
    saturation_effort=330.0,
    effort_limit={
        ".*_hip_pitch_joint": 320.0,
        ".*_hip_roll_joint": 330.0,
        ".*_hip_yaw_joint": 330.0,
        ".*_knee_joint": 330.0,
    },
    velocity_limit={
        ".*_hip_pitch_joint": 135.0,
        ".*_hip_roll_joint": 126.0,
        ".*_hip_yaw_joint": 126.0,
        ".*_knee_joint": 126.0,
    },
    stiffness={
        ".*_hip_pitch_joint": 200.0,
        ".*_hip_roll_joint": 250.0,
        ".*_hip_yaw_joint": 200.0,
        ".*_knee_joint": 250.0,
    },
    damping={".*": 20.0},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=10,
)

R1_FEET_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    saturation_effort=90.0,
    effort_limit=90.0,
    velocity_limit=157.0,
    stiffness={".*": 20.0},
    damping={".*": 5.0},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=10,
)

R1_ARMS_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
    saturation_effort=200.0,
    effort_limit={
        ".*_shoulder_pitch_joint": 130.0,
        ".*_shoulder_roll_joint": 200.0,
        ".*_shoulder_yaw_joint": 70.0,
        ".*_elbow_joint": 70.0,
        ".*_wrist_roll_joint": 50.0,
        ".*_wrist_pitch_joint": 50.0,
        ".*_wrist_yaw_joint": 36.0,
    },
    velocity_limit={
        ".*_shoulder_pitch_joint": 140.0,
        ".*_shoulder_roll_joint": 85.0,
        ".*_shoulder_yaw_joint": 117.0,
        ".*_elbow_joint": 117.0,
        ".*_wrist_roll_joint": 40.0,
        ".*_wrist_pitch_joint": 40.0,
        ".*_wrist_yaw_joint": 87.0,
    },
    stiffness={
        ".*_shoulder_pitch_joint": 20.0,
        ".*_shoulder_roll_joint": 36.0,
        ".*_shoulder_yaw_joint": 50.0,
        ".*_elbow_joint": 50.0,
        ".*_wrist_.*_joint": 50.0,
    },
    damping={
        ".*_shoulder_pitch_joint": 2.2,
        ".*_shoulder_roll_joint": 3.0,
        ".*_shoulder_yaw_joint": 0.4,
        ".*_elbow_joint": 1.0,
        ".*_wrist_roll_joint": 0.3,
        ".*_wrist_pitch_joint": 0.2,
        ".*_wrist_yaw_joint": 0.2,
    },
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=10,
)


@configclass
class R1PaceCfg(PaceCfg):
    """Identification parameters and real-data location for R1."""

    robot_name: str = "r1_real"
    data_dir: str = "r1_real/chirp_data.pt"
    joint_order: list[str] = R1_JOINT_ORDER
    bounds_params: torch.Tensor = torch.zeros((4 * len(R1_JOINT_ORDER) + 1, 2))

    # Absolute chirp command:
    # q_des_i(t) = center_i + amplitude_i * sin(chirp_phase(t) + phase_offset_i).
    # Collision-aware free-leg excitation. Hip/ankle roll and hip yaw move in
    # mirrored, outward-only ranges so the legs do not sweep through each
    # other. The chirp frequency law is unchanged. All trajectories start at
    # the initial position in R1_CFG (0 rad).
    trajectory_center: torch.Tensor = torch.tensor(
        [
            0.0,  # left hip pitch: [-0.20, 0.20]
            0.15,  # left hip roll: [0.00, 0.30], outward only
            -0.10,  # left hip yaw: [-0.20, 0.00]
            0.40,  # left knee: [0.00, 0.80]
            0.0,  # left ankle pitch: [-0.25, 0.25]
            0.125,  # left ankle roll: [0.00, 0.25], outward only
            0.0,  # right hip pitch: [-0.20, 0.20]
            -0.15,  # right hip roll: [-0.30, 0.00], outward only
            0.10,  # right hip yaw: [0.00, 0.20]
            0.40,  # right knee: [0.00, 0.80]
            0.0,  # right ankle pitch: [-0.25, 0.25]
            -0.125,  # right ankle roll: [-0.25, 0.00], outward only
        ],
        dtype=torch.float32,
    )
    trajectory_amplitude: torch.Tensor = torch.tensor(
        [
            0.20,
            -0.15,
            0.10,
            -0.40,
            0.25,
            -0.125,
            0.20,
            0.15,
            -0.10,
            -0.40,
            0.25,
            0.125,
        ],
        dtype=torch.float32,
    )
    trajectory_phase_offset: torch.Tensor = torch.tensor(
        [
            0.0,
            1.570796327,
            1.570796327,
            1.570796327,
            0.0,
            1.570796327,
            0.0,
            1.570796327,
            1.570796327,
            1.570796327,
            0.0,
            1.570796327,
        ],
        dtype=torch.float32,
    )

    def __post_init__(self):
        num_joints = len(self.joint_order)

        # [armature, viscous friction, Coulomb friction, encoder bias, delay]
        self.bounds_params[:num_joints, 0] = 1.0e-6
        self.bounds_params[:num_joints, 1] = 0.02
        self.bounds_params[num_joints : 2 * num_joints, 1] = 10.0
        self.bounds_params[2 * num_joints : 3 * num_joints, 1] = 0.5
        self.bounds_params[3 * num_joints : 4 * num_joints, 0] = -0.1
        self.bounds_params[3 * num_joints : 4 * num_joints, 1] = 0.1

        # Both knees start at 0 rad, which is also their URDF lower limit.
        # A negative encoder bias would initialize the simulated true joint
        # position below that limit (q_true = q_measured + bias).
        bias_start = 3 * num_joints
        left_knee_idx = self.joint_order.index("left_knee_joint")
        right_knee_idx = self.joint_order.index("right_knee_joint")
        self.bounds_params[bias_start + left_knee_idx, 0] = 0.0
        self.bounds_params[bias_start + right_knee_idx, 0] = 0.0

        self.bounds_params[4 * num_joints, 1] = 10.0


@configclass
class R1PaceSceneCfg(PaceSim2realSceneCfg):
    """R1 fixed-base scene used for actuator identification."""

    robot: ArticulationCfg = R1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "trunk": R1_TRUNK_PACE_ACTUATOR_CFG,
            "legs": R1_LEGS_PACE_ACTUATOR_CFG,
            "feet": R1_FEET_PACE_ACTUATOR_CFG,
            "arms": R1_ARMS_PACE_ACTUATOR_CFG,
        },
    )


@configclass
class R1PaceEnvCfg(PaceSim2realEnvCfg):
    scene: R1PaceSceneCfg = R1PaceSceneCfg()
    sim2real: PaceCfg = R1PaceCfg()

    def __post_init__(self):
        super().__post_init__()

        # This must equal the sampling period of the processed real-robot data.
        self.sim.dt = 0.0025
        self.decimation = 1
