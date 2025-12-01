from isaaclab_assets.robots.universal_robots import (
    UR10_LONG_SUCTION_CFG,
    UR10_SHORT_SUCTION_CFG,
)
from isaaclab.assets import RigidObjectCfg, SurfaceGripperCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from ... import mdp
from ...lift_env_cfg import LiftEnvCfg, RandomizedLiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.envs.mdp.actions.actions_cfg import SurfaceGripperBinaryActionCfg
@configclass
class UR10LiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Suction grippers currently require CPU simulation
        self.device = "cpu"
        downward_joints={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5707,
            "elbow_joint": 1.5707,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
        }
        # downward_joints={
        #     "shoulder_pan_joint": 3.14,
        #     "shoulder_lift_joint": -1.5707,
        #     "elbow_joint": -1.5707,
        #     "wrist_1_joint": -3.14,
        #     "wrist_2_joint": 1.5707963267948966,
        #     "wrist_3_joint": 0.0,
        # }
        custom_init = UR10_SHORT_SUCTION_CFG.init_state.replace(joint_pos=downward_joints)
        self.scene.robot = UR10_SHORT_SUCTION_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=custom_init,
        )
        # # Set UR10 as robot
        # self.scene.robot = UR10_SHORT_SUCTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set surface gripper: Ensure the SurfaceGripper prim has the required attributes
        self.scene.surface_gripper = SurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper",
            max_grip_distance=0.0075,
            shear_force_limit=5000.0,
            coaxial_force_limit=5000.0,
            retry_interval=0.05,
        )

        # Set actions for the specific robot type (ur10)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.159]),
        )
        # Set surface gripper action
        self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "ee_link"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                semantic_tags=[("class", "target_object")],
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.1585, 0.0, 0],
                    ),
                ),
                
            ],
        )