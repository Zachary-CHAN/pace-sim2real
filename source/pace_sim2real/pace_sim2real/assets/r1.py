# 复用IsaacLab的配置文件，做了一些适配本项目的修改
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 10 Hz -> rad/s
NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

# Armature (等效转动惯量 I)
ARMATURE_13715 = 0.1986  # hip_pitch, waist_roll
ARMATURE_10020_2 = 0.2748  # hip_roll_yaw_knee, waist_pitch
ARMATURE_8112 = 0.0483  # ankle
ARMATURE_10020_1 = 0.0698  # waist_yaw
ARMATURE_8116 = 0.0638  # shoulder_pitch
ARMATURE_4315 = 0.0324  # shoulder_yaw, elbow
ARMATURE_4310_2 = 0.0246  # wrist_yaw

ARMATURE_86_3 = 0.02  # shoulder_roll (estimated value)
ARMATURE_PF_L28 = 0.001  # electric cylinder substitution


# # Stiffness: K = I * wn^2
# STIFFNESS_8112 = ARMATURE_8112 * NATURAL_FREQ**2
# STIFFNESS_13715 = ARMATURE_13715 * NATURAL_FREQ**2
# STIFFNESS_10020_1 = ARMATURE_10020_1 * NATURAL_FREQ**2
# STIFFNESS_10020_2 = ARMATURE_10020_2 * NATURAL_FREQ**2
# STIFFNESS_8116 = ARMATURE_8116 * NATURAL_FREQ**2
# STIFFNESS_86_3 = ARMATURE_86_3 * NATURAL_FREQ**2
# STIFFNESS_4315 = ARMATURE_4315 * NATURAL_FREQ**2
# STIFFNESS_4310_2 = ARMATURE_4310_2 * NATURAL_FREQ**2
# STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2

# # Damping: D = 2 * zeta * I * wn
# DAMPING_8112 = 2.0 * DAMPING_RATIO * ARMATURE_8112 * NATURAL_FREQ
# DAMPING_13715 = 2.0 * DAMPING_RATIO * ARMATURE_13715 * NATURAL_FREQ
# DAMPING_10020_1 = 2.0 * DAMPING_RATIO * ARMATURE_10020_1 * NATURAL_FREQ
# DAMPING_10020_2 = 2.0 * DAMPING_RATIO * ARMATURE_10020_2 * NATURAL_FREQ
# DAMPING_8116 = 2.0 * DAMPING_RATIO * ARMATURE_8116 * NATURAL_FREQ
# DAMPING_86_3 = 2.0 * DAMPING_RATIO * ARMATURE_86_3 * NATURAL_FREQ
# DAMPING_4315 = 2.0 * DAMPING_RATIO * ARMATURE_4315 * NATURAL_FREQ
# DAMPING_4310_2 = 2.0 * DAMPING_RATIO * ARMATURE_4310_2 * NATURAL_FREQ
# DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ

# kp = tau / v * 1.5    Kd = Kp * 0.064
STIFFNESS_8112 = 8.59
STIFFNESS_13715 = 35.55
STIFFNESS_10020_1 = 16.07
STIFFNESS_10020_2 = 39.28
STIFFNESS_8116 = 14
STIFFNESS_86_3 = 35.29
STIFFNESS_4315 = 9
STIFFNESS_4310_2 = 6.2
STIFFNESS_5020 = 18.75

DAMPING_8112 = 0.54
DAMPING_13715 = 2.2752
DAMPING_10020_1 = 1.02
DAMPING_10020_2 = 2.51
DAMPING_8116 = 0.896
DAMPING_86_3 = 2.258
DAMPING_4315 = 0.576
DAMPING_4310_2 = 0.384
DAMPING_5020 = 1.2

R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/czq/HNU_R1/R1/R1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # PACE identifies joint dynamics on a test stand.  Fixing the root
            # prevents base motion/contact dynamics from contaminating the fit.
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.95),
    #     joint_pos={
    #         ".*_hip_pitch_joint": -0.52,
    #         "left_hip_roll_joint": -0.03,
    #         "right_hip_roll_joint": 0.03,
    #         "left_hip_yaw_joint": -0.27,
    #         "right_hip_yaw_joint": 0.27,
    #         ".*_knee_joint": 0.67,
    #         ".*_ankle_pitch_joint": -0.23,
    #         "left_ankle_roll_joint": 0.0,
    #         "right_ankle_roll_joint": 0.0,
    #         "waist_yaw_joint": 0.0,
    #         "waist_roll_joint": 0.0,
    #         "waist_pitch_joint": 0.0,
    #         ".*_shoulder_pitch_joint": 0.0,
    #         ".*_shoulder_roll_joint": 0.0,
    #         ".*_shoulder_yaw_joint": 0.0,
    #         ".*_elbow_joint": 0.0,  # -1.57 保持90度弯曲
    #         ".*_wrist_.*_joint": 0.0,
    #     },
    #     joint_vel={".*": 0.0},
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": -1.57,  # -1.57 保持90度弯曲
            ".*_wrist_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "trunk": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*",
            ],
            effort_limit=300.0,
            velocity_limit=10.0,
            stiffness=15000.0,
            damping=300.0,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 320.0,
                ".*_hip_roll_joint": 330.0,
                ".*_hip_yaw_joint": 330.0,
                ".*_knee_joint": 330.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 135,
                ".*_hip_roll_joint": 126,
                ".*_hip_yaw_joint": 126,
                ".*_knee_joint": 126,
            },
            stiffness={
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_roll_joint": 250.0,
                ".*_hip_yaw_joint": 200.0,
                ".*_knee_joint": 250.0,
            },
            damping={
                ".*_hip_pitch_joint": 20,
                ".*_hip_roll_joint": 20,
                ".*_hip_yaw_joint": 20,
                ".*_knee_joint": 20,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_13715,
                ".*_hip_roll_joint": ARMATURE_10020_2,
                ".*_hip_yaw_joint": ARMATURE_10020_2,
                ".*_knee_joint": ARMATURE_10020_2,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=90.0,
            velocity_limit_sim=157.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20,
            damping=5,
            armature=ARMATURE_8112,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 130.0,
                ".*_shoulder_roll_joint": 200.0,
                ".*_shoulder_yaw_joint": 70.0,
                ".*_elbow_joint": 70.0,
                ".*_wrist_roll_joint": 50.0,
                ".*_wrist_pitch_joint": 50.0,
                ".*_wrist_yaw_joint": 36.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 140.0,
                ".*_shoulder_roll_joint": 85.0,
                ".*_shoulder_yaw_joint": 117.0,
                ".*_elbow_joint": 117.0,
                ".*_wrist_roll_joint": 40.0,
                ".*_wrist_pitch_joint": 40.0,
                ".*_wrist_yaw_joint": 87.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 20,
                ".*_shoulder_roll_joint": 36,
                ".*_shoulder_yaw_joint": 50,
                ".*_elbow_joint": 50,
                ".*_wrist_roll_joint": 50,
                ".*_wrist_pitch_joint": 50,
                ".*_wrist_yaw_joint": 50,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.2,
                ".*_shoulder_roll_joint": 3,
                ".*_shoulder_yaw_joint": 0.4,
                ".*_elbow_joint": 1,
                ".*_wrist_roll_joint": 0.3,
                ".*_wrist_pitch_joint": 0.2,
                ".*_wrist_yaw_joint": 0.2,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_8116,
                ".*_shoulder_roll_joint": ARMATURE_86_3,
                ".*_shoulder_yaw_joint": ARMATURE_4315,
                ".*_elbow_joint": ARMATURE_4315,
                ".*_wrist_roll_joint": ARMATURE_PF_L28,
                ".*_wrist_pitch_joint": ARMATURE_PF_L28,
                ".*_wrist_yaw_joint": ARMATURE_4310_2,
            },
        ),
    },
)
