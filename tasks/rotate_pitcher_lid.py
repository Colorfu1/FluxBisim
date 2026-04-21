import os

import numpy as np
import rootutils
from isaacsim.core.prims import XFormPrim

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402


def get_prim_world_pos(world, prim_path: str):
    """Return the world position `(x, y, z)` of a prim."""
    prim_view = XFormPrim(
        prim_paths_expr=prim_path,
        name=f"tmp_{prim_path.split('/')[-1]}",
    )
    pos, _ = prim_view.get_world_poses()
    return np.array(pos[0])


class RotatePitcherLid:
    PITCHER_RANDOMIZATION_RADIUS = 0.005
    PITCHER_RANDOMIZATION_RADIUS_NOISE = 0.01
    LID_RANDOMIZATION_RADIUS = 0.02
    LID_RANDOMIZATION_RADIUS_NOISE = 0.04

    def __init__(self):
        self.pitcher_name = os.environ.get("PITCHER_NAME", "pitcher")
        self.lid_name = os.environ.get("LID_NAME", "pitcherlid")
        self.table_name = os.environ.get("TABLE_NAME", "kitchen_table")

        self.objects = [
            BaseObject(self.pitcher_name),
            BaseObject(self.lid_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(lid_name=self.lid_name, pitcher_name=self.pitcher_name)

    def _generate_point_in_circle(self, origin_x, origin_y, origin_z, radius=0.01):
        """Sample a random point inside a circle."""
        theta = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        x = origin_x + r * np.cos(theta)
        y = origin_y + r * np.sin(theta)
        return [x, y, origin_z]

    def randomize_objects_position(self):
        """Randomize the pitcher and lid around their initial poses."""
        for obj in self.objects:
            if obj.name == self.pitcher_name:
                default_position = self._generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.PITCHER_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(positions=np.array([default_position]))
            elif obj.name == self.lid_name:
                default_position = self._generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.LID_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.pitcher_name or obj.name == self.lid_name:
                print(
                    "obj.name:",
                    obj.name,
                    "new position:",
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    PLACE_THRESHOLD = 0.015
    ORIENTATION_THRESHOLD_DEG = 5.0
    TARGET_LID_QUAT = np.array([0.77, 0.00303, -0.00313, 0.64])
    RIGHT_EE_OFFSET = np.array([-0.01, 0.0, -0.17])
    LEFT_EE_OFFSET = np.array([0.413, 0.03, -0.14])
    GRIPPER_THRESHOLD = 0.2

    def __init__(self, lid_name, pitcher_name):
        self.lid_name = lid_name
        self.pitcher_name = pitcher_name

    def check_gripper(self, env):
        """Check whether the gripper is holding the object."""
        current_ee_pos = env.robot.right_manipulator.end_effector.get_world_pose()[
            0
        ] + np.array([0.0, 0.0, -0.13])
        object_pos = env.world.scene.get_object(self.lid_name).get_world_poses()[0][0]

        ee_distance = np.linalg.norm(current_ee_pos - object_pos)

        if ee_distance > self.GRIPPER_THRESHOLD:
            print("The gripper is not holding the object.")
            return False
        else:
            return True

    def check(self, env):
        """Check whether the lid is at the pitcher opening with the desired pose."""
        lid = env.world.scene.get_object(self.lid_name)
        pitcher = env.world.scene.get_object(self.pitcher_name)
        if lid is None or pitcher is None:
            raise ValueError("Lid or pitcher object does not exist.")

        lid_pos_raw, lid_orient_raw = lid.get_world_poses()
        lid_pos = np.array(lid_pos_raw[0])
        lid_orient = np.array(lid_orient_raw[0])
        pitcher_pos = get_prim_world_pos(
            env.world, f"/World/{self.pitcher_name}/pitcher_base/Visuals"
        )

        distance_xy = np.linalg.norm(lid_pos[:2] - pitcher_pos[:2])

        if np.linalg.norm(lid_orient) > 0:
            lid_q = lid_orient / np.linalg.norm(lid_orient)
            target_q = self.TARGET_LID_QUAT / np.linalg.norm(self.TARGET_LID_QUAT)
            # Use absolute dot product to treat q and -q as the same rotation.
            dot = float(np.clip(np.abs(np.dot(lid_q, target_q)), -1.0, 1.0))
            angle_diff_deg = np.degrees(2.0 * np.arccos(dot))
        else:
            angle_diff_deg = 180.0

        return (
            distance_xy <= self.PLACE_THRESHOLD
            and angle_diff_deg < self.ORIENTATION_THRESHOLD_DEG
        )


# class Checker:
#     def __init__(self, lid_name, target_name):
#         self.obj_name = lid_name
#         self.target_name = target_name
#         self.reset_episode()

#     def reset_episode(self) -> None:
#         """Reset all TCP and grasp state for a new episode."""
#         # Three sub-task scores:
#         # 1. The left gripper holds an object for long enough.
#         # 2. The right gripper holds the lid for long enough.
#         # 3. The lid is placed onto the pitcher.
#         self.score_left_grab_object: float = 0.0
#         self.score_right_grab_lid: float = 0.0
#         self.score_lid_on_pitcher: float = 0.0

#         # Current frame TCP and historical max TCP in percentage.
#         self.current_tcp: float = 0.0
#         self.max_tcp: float = 0.0

#         # Grasp confirmation state.
#         self.left_grab_counter: int = 0
#         self.right_grab_counter: int = 0
#         self.left_grab_confirmed: bool = False
#         self.right_grab_confirmed: bool = False

#         # Required number of consecutive frames to confirm a grasp.
#         self.left_grab_confirm_frames: int = 150
#         self.right_grab_confirm_frames: int = 150

#     def check_gripper(self, robot, env):
#         """Check whether the gripper is holding the target object."""
#         threshold = 0.12  # Grasp distance threshold in meters.

#         if robot == 1:
#             current_ee_pos = np.array(env.robot.right_manipulator.end_effector.get_world_pose()[0] + np.array([-0.01, 0.0, -0.17]))
#             object_position = np.array(env.world.scene.get_object(self.obj_name).get_world_poses()[0][0])
#         else:
#             current_ee_pos = np.array(env.robot.left_manipulator.end_effector.get_world_pose()[0] + np.array([0.413, 0.03, -0.14]))
#             object_position = get_prim_world_pos(env.world, "/World/pitcher/pitcher_base/Visuals")
#         # print(f"{robot} current_ee_pos: {current_ee_pos}")
#         # print(f"{robot} object_position: {object_position}")

#         # Measure the distance between the end effector and the object.
#         ee_distance = np.linalg.norm(current_ee_pos - object_position)

#         if ee_distance > threshold:
#             print(f"Robot {robot} gripper is not holding the object: offset {ee_distance:.3f} m")
#             return False
#         else:
#             # print(f"Robot {robot} gripper is holding the object: offset {ee_distance:.3f} m")
#             return True

#     def check_task(self, env):
#         """Check whether the object is at the target position."""
#         obj = env.world.scene.get_object(self.obj_name)
#         target = env.world.scene.get_object(self.target_name)
#         if obj is None or target is None:
#             raise ValueError(f"Object or target '{self.obj_name}' or '{self.target_name}' does not exist.")

#         obj_pos = obj.get_world_poses()[0][0]
#         target_pos = get_prim_world_pos(env.world, "/World/pitcher/pitcher_base/Visuals")
#         print(f"obj_pos: {obj_pos}")
#         print(f"target_pos: {target_pos}")

#         distance = np.linalg.norm(obj_pos[:-1] - target_pos[:-1])

#         return distance < 0.04  # The position threshold can be tuned if needed.

#     def update_TCP_per_step(self, env) -> None:
#         """
#         Update the three sub-task scores and TCP on every inference step.

#         The TCP is computed from whether the left arm is grasping an object,
#         whether the right arm is grasping the lid, and whether the lid is
#         aligned with the pitcher opening.
#         """

#         # ---------- 1. Check whether the left gripper is holding an object ----------
#         left_gripper_joints = env.robot.left_manipulator.gripper.get_joint_positions()
#         left_gripper_width = float(np.mean(np.abs(left_gripper_joints)))
#         # A half-closed gripper usually indicates a stable grasp.
#         left_grip_min_threshold = 0.010
#         left_grip_max_threshold = 0.030
#         is_left_gripping = left_grip_min_threshold < left_gripper_width < left_grip_max_threshold

#         if is_left_gripping:
#             self.left_grab_counter += 1
#         elif not self.left_grab_confirmed:
#             # Reset the counter until the grasp is confirmed.
#             self.left_grab_counter = 0

#         if self.left_grab_counter >= self.left_grab_confirm_frames:
#             self.left_grab_confirmed = True

#         self.score_left_grab_object = 1.0 if self.left_grab_confirmed else 0.0

#         # ---------- 2. Check whether the right gripper is holding the lid ----------
#         right_gripper_joints = env.robot.right_manipulator.gripper.get_joint_positions()
#         right_gripper_width = float(np.mean(np.abs(right_gripper_joints)))
#         right_grip_min_threshold = 0.010
#         right_grip_max_threshold = 0.030
#         is_right_gripping = right_grip_min_threshold < right_gripper_width < right_grip_max_threshold

#         if is_right_gripping:
#             self.right_grab_counter += 1
#         elif not self.right_grab_confirmed:
#             self.right_grab_counter = 0

#         if self.right_grab_counter >= self.right_grab_confirm_frames:
#             self.right_grab_confirmed = True

#         self.score_right_grab_lid = 1.0 if self.right_grab_confirmed else 0.0

#         # ---------- 3. Check whether the lid is aligned with the pitcher ----------
#         self.score_lid_on_pitcher = 0.0
#         if self.score_right_grab_lid > 0.5:
#             lid_obj = env.world.scene.get_object(self.obj_name)
#             pitcher_obj = env.world.scene.get_object(self.target_name)
#             if lid_obj is None or pitcher_obj is None:
#                 raise ValueError(
#                     f"Lid object '{self.obj_name}' or pitcher object '{self.target_name}' does not exist."
#                 )

#             lid_pos = np.array(lid_obj.get_world_poses()[0][0])
#             pitcher_top_pos = get_prim_world_pos(
#                 env.world,
#                 "/World/pitcher/pitcher_base/Visuals"
#             )

#             lid_pitcher_dist = np.linalg.norm(lid_pos[:-1] - pitcher_top_pos[:-1])
#             lid_on_pitcher_thresh = 0.04  # Keep this aligned with `check_task`.

#             # Consider the lid placed when it is close to the pitcher opening.
#             self.score_lid_on_pitcher = 1.0 if lid_pitcher_dist < lid_on_pitcher_thresh else 0.0

#         # ---------- Aggregate the current TCP and track its maximum ----------
#         tcp = (
#             self.score_left_grab_object
#             + self.score_right_grab_lid
#             + self.score_lid_on_pitcher
#         ) / 3.0
#         self.current_tcp = tcp * 100.0

#         if self.current_tcp > self.max_tcp:
#             self.max_tcp = self.current_tcp

#     def get_TCP(self, use_max: bool = True) -> float:
#         """
#         Return the task completion percentage at the end of an episode.

#         When `use_max` is `True`, this returns the maximum TCP reached during
#         the episode. Otherwise it returns the TCP from the last frame.
#         The internal state is reset afterwards for the next episode.
#         """
#         tcp_value = self.max_tcp if use_max else self.current_tcp
#         self.reset_episode()
#         return tcp_value
