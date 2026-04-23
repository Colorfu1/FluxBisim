import os

import numpy as np
import yaml
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from scipy.spatial.transform import Rotation as R


class BaseRobot:
    def __init__(self, name):
        self.robot_cfg = self._load_robot_config()
        self.name = name
        self.base_name = os.environ.get("BASE_NAME", "base_kitchen")
        self.left_manipulator = None
        self.right_manipulator = None

    def _load_robot_config(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "cfg", "robots.yaml")
        with open(cfg_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config

    def init_robot(self, world):
        """Initialize the robot and create its ``XFormPrim`` instances."""
        if self.robot_cfg is None:
            raise ValueError(f"Configuration for '{self.name}' was not found.")

        # Add the robot base.
        if self.base_name in self.robot_cfg:
            base_cfg = self.robot_cfg[self.base_name]
            add_reference_to_stage(
                usd_path=base_cfg["usd_path"], prim_path=f"/World/{self.name}_base"
            )
            # Support base position, orientation, and relative delta.
            base_position = np.array(base_cfg.get("position", [0.0, 0.0, 0.0]))
            base_orientation = np.array(
                base_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0])
            )
            _ = XFormPrim(
                prim_paths_expr=f"/World/{self.name}_base",
                positions=np.array([base_position]),
                orientations=np.array([base_orientation]),
                visibilities=np.array([[0.32809, 0.37396, 0.69]]),
            )

        if "left_arm" in self.robot_cfg:
            left_arm_cfg = self.robot_cfg["left_arm"]
            add_reference_to_stage(
                usd_path=left_arm_cfg["usd_path"],
                prim_path=f"/World/{self.name}_left_arm",
            )
            # Compute the arm pose in the world frame.
            left_arm_relative_pos = np.array(left_arm_cfg["position"])
            left_arm_relative_quat = np.array(left_arm_cfg["orientation"])

            # Transform the relative pose into the world frame.
            base_rot = R.from_quat(
                base_orientation[[1, 2, 3, 0]]
            )  # Convert [w,x,y,z] to scipy order [x,y,z,w].
            left_arm_world_pos = base_position + base_rot.apply(left_arm_relative_pos)
            left_arm_world_quat = (
                base_rot * R.from_quat(left_arm_relative_quat[[1, 2, 3, 0]])
            ).as_quat()[[3, 0, 1, 2]]  # Convert back to [w,x,y,z] order.

            _ = XFormPrim(
                prim_paths_expr=f"/World/{self.name}_left_arm",
                positions=np.array([left_arm_world_pos]),
                orientations=np.array([left_arm_world_quat]),
                visibilities=np.array([[0.32809, 0.37396, 0.69]]),
            )
            left_griper = ParallelGripper(
                end_effector_prim_path=f"/World/{self.name}_left_arm/gripper_base",
                joint_prim_names=["joint7", "joint8"],
                joint_opened_positions=np.array([0.035, -0.035]),
                joint_closed_positions=np.array([0.0, -0.0]),
                action_deltas=np.array([0.035, -0.035]),
            )
            self.left_manipulator = SingleManipulator(
                prim_path=f"/World/{self.name}_left_arm",
                name=f"{self.name}_left_arm",
                end_effector_prim_name="gripper_base",
                gripper=left_griper,
            )
            joints_default_positions = np.zeros(8)
            joints_default_positions[6] = 0.035
            joints_default_positions[7] = -0.035
            self.left_manipulator.set_joints_default_state(
                positions=joints_default_positions
            )

        if "right_arm" in self.robot_cfg:
            right_arm_cfg = self.robot_cfg["right_arm"]
            add_reference_to_stage(
                usd_path=right_arm_cfg["usd_path"],
                prim_path=f"/World/{self.name}_right_arm",
            )
            # Compute the arm pose in the world frame.
            right_arm_relative_pos = np.array(right_arm_cfg["position"])
            right_arm_relative_quat = np.array(right_arm_cfg["orientation"])

            # Transform the relative pose into the world frame.
            base_rot = R.from_quat(
                base_orientation[[1, 2, 3, 0]]
            )  # Convert [w,x,y,z] to scipy order [x,y,z,w].
            right_arm_world_pos = base_position + base_rot.apply(right_arm_relative_pos)
            right_arm_world_quat = (
                base_rot * R.from_quat(right_arm_relative_quat[[1, 2, 3, 0]])
            ).as_quat()[[3, 0, 1, 2]]  # Convert back to [w,x,y,z] order.

            _ = XFormPrim(
                prim_paths_expr=f"/World/{self.name}_right_arm",
                positions=np.array([right_arm_world_pos]),
                orientations=np.array([right_arm_world_quat]),
                visibilities=np.array([[0.32809, 0.37396, 0.69]]),
            )
            right_griper = ParallelGripper(
                end_effector_prim_path=f"/World/{self.name}_right_arm/gripper_base",
                joint_prim_names=["joint7", "joint8"],
                joint_opened_positions=np.array([0.035, -0.035]),
                joint_closed_positions=np.array([0.0, -0.0]),
                action_deltas=np.array([0.035, -0.035]),
            )
            self.right_manipulator = SingleManipulator(
                prim_path=f"/World/{self.name}_right_arm",
                name=f"{self.name}_right_arm",
                end_effector_prim_name="gripper_base",
                gripper=right_griper,
            )
            joints_default_positions = np.zeros(8)
            joints_default_positions[6] = 0
            joints_default_positions[7] = 0
            self.right_manipulator.set_joints_default_state(
                positions=joints_default_positions
            )

        world.scene.add(self.left_manipulator)
        world.scene.add(self.right_manipulator)

    def reset(self):
        noise = np.random.uniform(-0.1, 0.1, 6)
        left_joints = np.array([-0.2, 0.3491, -0.3491, 0.0, 0.1, 0.0, 0.035, -0.035])
        left_joints[:6] += noise
        self.left_manipulator.set_joints_default_state(positions=left_joints)
        self.left_manipulator.post_reset()
        self.left_manipulator.gripper.set_joint_positions(
            self.left_manipulator.gripper.joint_opened_positions
        )

        right_joints = np.array([0.2, 0.3491, -0.3491, 0.0, 0.1, 0.0, 0.035, -0.035])
        right_joints[:6] += noise
        self.right_manipulator.set_joints_default_state(positions=right_joints)
        self.right_manipulator.post_reset()
        self.right_manipulator.gripper.set_joint_positions(
            self.right_manipulator.gripper.joint_opened_positions
        )

    def apply_action(self, action):
        if len(action) > 1:
            if action[0] is not None:
                self.left_manipulator.apply_action(action[0])
            if action[1] is not None:
                self.right_manipulator.apply_action(action[1])
        else:
            self.right_manipulator.apply_action(action[0])
