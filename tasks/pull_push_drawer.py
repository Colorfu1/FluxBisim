import os

import numpy as np
import rootutils
from isaacsim.core.prims import XFormPrim

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402


def get_prim_world_pos(world, prim_path: str):
    """Return the world position (x, y, z) for the given prim path."""
    prim_view = XFormPrim(
        prim_paths_expr=prim_path,
        name=f"tmp_{prim_path.split('/')[-1]}",
    )
    pos, _ = prim_view.get_world_poses()
    return np.array(pos[0])


class PullPushDrawer:
    RANDOMIZATION_RADIUS = 0.01
    RANDOMIZATION_RADIUS_NOISE = 0.02

    def __init__(self):
        self.object_name = os.environ.get("OBJECT_NAME", "apple_drawer")
        self.drawer_name = os.environ.get("DRAWER_NAME", "drawer")
        self.table_name = os.environ.get("TABLE_NAME", "kitchen_table")

        self.objects = [
            BaseObject(self.object_name),
            BaseObject(self.drawer_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(
            object_name=self.object_name, drawer_name=self.drawer_name
        )

    def _generate_point_in_circle(self, origin_x, origin_y, origin_z, radius=0.01):
        """Sample a random point inside a circle."""
        theta = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        x = origin_x + r * np.cos(theta)
        y = origin_y + r * np.sin(theta)
        return [x, y, origin_z]

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.object_name or obj.name == self.drawer_name:
                default_position = self._generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.object_name or obj.name == self.drawer_name:
                print(
                    "obj.name:",
                    obj.name,
                    "new position:",
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    POSITION_THRESHOLD = 0.05  # Maximum distance to drawer target position.
    MIN_OPEN_DIST = 0.10  # Minimum drawer opening distance.
    MAX_CLOSED_DIST = 0.02  # Maximum drawer closing distance.
    GRIPPER_THRESHOLD = 0.2  # Maximum EE-to-object distance for a valid grasp.
    DRAWER_OFFSET = 0.04646722  # Baseline offset for drawer distance measurement.

    def __init__(self, object_name, drawer_name):
        """Initialize the checker for the pull-push drawer task."""
        self.object_name = object_name
        self.drawer_name = drawer_name
        self.max_drawer_dist = 0.0

    def check_gripper(self, env):
        """Check whether the gripper is holding the object."""
        current_ee_pos = env.robot.left_manipulator.end_effector.get_world_pose()[
            0
        ] + np.array([0.0, 0.0, -0.13])
        object_pos = env.world.scene.get_object(self.object_name).get_world_poses()[0][
            0
        ]

        # Measure the distance between the end effector and the object.
        ee_distance = np.linalg.norm(current_ee_pos - object_pos)

        if ee_distance > self.GRIPPER_THRESHOLD:
            print("The gripper is not holding the object.")
            return False
        else:
            return True

    def _get_drawer_distance(self, env):
        """Calculate the current drawer opening distance."""
        drawer_front_pos = get_prim_world_pos(env.world, "/World/drawer/E_drawer2_7")
        target_part_pos = get_prim_world_pos(
            env.world, "/World/drawer/E_body_1/E_part4_5"
        )
        return np.linalg.norm(drawer_front_pos - target_part_pos) - self.DRAWER_OFFSET

    def calculate_open_distance(self, env):
        distance = self._get_drawer_distance(env)
        if distance > self.max_drawer_dist:
            self.max_drawer_dist = distance

    def check_drawer_open(self):
        return self.max_drawer_dist > self.MIN_OPEN_DIST

    def check_drawer_closed(self, env):
        return self._get_drawer_distance(env) < self.MAX_CLOSED_DIST

    def check(self, env):
        """Check whether the task is complete."""
        object = env.world.scene.get_object(self.object_name)
        drawer = env.world.scene.get_object(self.drawer_name)
        if object is None or drawer is None:
            raise ValueError("Object or drawer does not exist.")

        object_pos = object.get_world_poses()[0][0]
        drawer_pos = get_prim_world_pos(
            env.world, "/World/drawer/E_drawer2_7"
        ) + np.array([-0.063, 0.083, 0.0018])

        distance_xy = np.linalg.norm(object_pos[:2] - drawer_pos[:2])

        return distance_xy <= self.POSITION_THRESHOLD and self.check_drawer_closed(env)


# class Checker:
#     """Task completion checker."""
#     POSITION_THRESHOLD = 0.05  # Maximum distance to drawer target position.
#     MIN_OPEN_DIST = 0.10  # Minimum drawer opening distance.
#     GRIPPER_THRESHOLD = 0.2  # Maximum EE-to-object distance for a valid grasp.
#     DRAWER_BASELINE_OFFSET = 0.04646722  # Baseline offset for drawer distance measurement.
#     LEFT_GRIP_MIN = 0.010  # Minimum gripper width for a valid left grasp.
#     LEFT_GRIP_MAX = 0.030  # Maximum gripper width for a valid left grasp.
#     LEFT_CONFIRM_FRAMES = 150  # Frames required to confirm left grasp.
#     RIGHT_GRIP_MIN = 0.010  # Minimum gripper width for a valid right grasp.
#     RIGHT_GRIP_MAX = 0.030  # Maximum gripper width for a valid right grasp.
#     RIGHT_CONFIRM_FRAMES = 90  # Frames required to confirm right grasp.

#     def __init__(self, chef_name, drawer_name):
#         """Initialize the checker for the pull-push drawer task."""
#         self.chef_name = chef_name
#         self.drawer_name = drawer_name
#         self.max_drawer_dist = 0.0
#         self.reset_episode()

#     def reset_episode(self) -> None:
#         """Reset all TCP and grasp state for a new episode."""
#         self.score_left_grab_handle = 0.0
#         self.score_drawer_opened = 0.0
#         self.score_right_grab_drawer = 0.0
#         self.score_object_in_drawer = 0.0

#         self.current_tcp = 0.0
#         self.max_tcp = 0.0

#         self.left_grab_counter = 0
#         self.right_grab_counter = 0
#         self.left_grab_confirmed = False
#         self.right_grab_confirmed = False

#         self.drawer_open_confirmed = False
#         self.max_drawer_dist = 0.0

#     def check_gripper(self, env):
#         """Check whether the gripper is holding the object."""
#         current_ee_pos = np.array(env.robot.left_manipulator.end_effector.get_world_pose()[0] + np.array([0.0, 0.0, -0.13]))
#         object_position = np.array(env.world.scene.get_object(self.chef_name).get_world_poses()[0][0])

#         ee_distance = np.linalg.norm(current_ee_pos - object_position)

#         if ee_distance > self.GRIPPER_THRESHOLD:
#             print(f"The gripper is not holding the object: offset {ee_distance:.3f} m")
#             return False
#         else:
#             return True

#     def check_drawer(self, env):
#         """Check whether the drawer is open."""
#         drawer_front_pos = get_prim_world_pos(env.world, "/World/drawer/E_drawer2_7")
#         target_part_pos = get_prim_world_pos(env.world, "/World/drawer/E_body_1/E_part4_5")
#         distance = np.linalg.norm(drawer_front_pos - target_part_pos) - self.DRAWER_BASELINE_OFFSET

#         if distance > self.max_drawer_dist:
#             self.max_drawer_dist = distance
#         print("Drawer front to reference distance:", distance)

#         return distance > -0.001

#     def check(self, env):
#         """Check whether the task is complete."""
#         obj = env.world.scene.get_object(self.chef_name)
#         target = env.world.scene.get_object(self.drawer_name)
#         if obj is None or target is None:
#             raise ValueError(f"Chef or drawer object does not exist.")

#         obj_pos = obj.get_world_poses()[0][0]
#         drawer_front_pos_offset = get_prim_world_pos(env.world, "/World/drawer/E_drawer2_7") + np.array([-0.063, 0.083, 0.0018])

#         pos_distance = np.linalg.norm(obj_pos[:-1] - drawer_front_pos_offset[:-1])
#         in_target_range = pos_distance < self.POSITION_THRESHOLD

#         opened_enough = self.max_drawer_dist >= self.MIN_OPEN_DIST

#         # print(
#         #     f"[check] pos_distance={pos_distance:.4f}, "
#         #     f"in_target_range={in_target_range}, "
#         #     f"max_drawer_dist={self.max_drawer_dist:.4f}, "
#         #     f"opened_enough={opened_enough}"
#         # )

#         return in_target_range and opened_enough

#     def check_SR(self, env):
#         """Check whether the object is near the drawer target position."""
#         obj = env.world.scene.get_object(self.chef_name)
#         target = env.world.scene.get_object(self.drawer_name)
#         if obj is None or target is None:
#             raise ValueError(f"Chef or drawer object does not exist.")

#         obj_pos = obj.get_world_poses()[0][0]
#         drawer_front_pos_offset = get_prim_world_pos(env.world, "/World/drawer/E_drawer2_7") + np.array([-0.063, 0.083, 0.0018])

#         pos_distance = np.linalg.norm(obj_pos[:-1] - drawer_front_pos_offset[:-1])
#         in_target_range = pos_distance < self.POSITION_THRESHOLD

#         # print(
#         #     f"[check_SR] pos_distance={pos_distance:.4f}, "
#         #     f"in_target_range={in_target_range}, "
#         # )

#         return in_target_range

#     def update_TCP_per_step(self, env) -> None:
#         """Update the four sub-task scores and TCP on every inference step."""

#         # ---------- 1. Left arm grasp handle ----------
#         left_gripper_joints = env.robot.left_manipulator.gripper.get_joint_positions()
#         left_gripper_width = float(np.mean(np.abs(left_gripper_joints)))

#         is_left_gripping = self.LEFT_GRIP_MIN < left_gripper_width < self.LEFT_GRIP_MAX

#         if is_left_gripping:
#             self.left_grab_counter += 1
#         elif not self.left_grab_confirmed:
#             self.left_grab_counter = 0

#         if self.left_grab_counter >= self.LEFT_CONFIRM_FRAMES:
#             self.left_grab_confirmed = True

#         self.score_left_grab_handle = 1.0 if self.left_grab_confirmed else 0.0

#         # ---------- 2. Right arm grasp drawer ----------
#         right_gripper_joints = env.robot.right_manipulator.gripper.get_joint_positions()
#         right_gripper_width = float(np.mean(np.abs(right_gripper_joints)))

#         is_right_gripping = self.RIGHT_GRIP_MIN < right_gripper_width < self.RIGHT_GRIP_MAX

#         if is_right_gripping:
#             self.right_grab_counter += 1
#         elif not self.right_grab_confirmed:
#             self.right_grab_counter = 0

#         if self.right_grab_counter >= self.RIGHT_CONFIRM_FRAMES:
#             self.right_grab_confirmed = True

#         self.score_right_grab_drawer = 1.0 if self.right_grab_confirmed else 0.0

#         # ---------- 3. Drawer opening ----------
#         drawer_front_pos = get_prim_world_pos(env.world, "/World/drawer/E_drawer2_7")
#         target_part_pos = get_prim_world_pos(env.world, "/World/drawer/E_body_1/E_part4_5")
#         distance = np.linalg.norm(drawer_front_pos - target_part_pos) - self.DRAWER_BASELINE_OFFSET

#         if distance > self.max_drawer_dist:
#             self.max_drawer_dist = distance

#         self.drawer_open_confirmed = self.max_drawer_dist >= self.MIN_OPEN_DIST
#         self.score_drawer_opened = 1.0 if (self.drawer_open_confirmed and self.right_grab_confirmed) else 0.0

#         # ---------- 4. Object placed in drawer ----------
#         self.score_object_in_drawer = 0.0
#         if self.drawer_open_confirmed:
#             obj = env.world.scene.get_object(self.chef_name)
#             drawer = env.world.scene.get_object(self.drawer_name)
#             if obj is None or drawer is None:
#                 raise ValueError(f"Chef or drawer object does not exist.")

#             drawer_front_pos_offset = get_prim_world_pos(
#                 env.world,
#                 "/World/drawer/E_drawer2_7"
#             ) + np.array([-0.063, 0.083, 0.0018])

#             obj_pos = obj.get_world_poses()[0][0]
#             pos_distance = np.linalg.norm(obj_pos[:-1] - drawer_front_pos_offset[:-1])

#             self.score_object_in_drawer = 1.0 if (pos_distance < self.POSITION_THRESHOLD and self.drawer_open_confirmed) else 0.0

#         # ---------- Aggregate TCP ----------
#         tcp = (
#             self.score_left_grab_handle
#             + self.score_drawer_opened
#             + self.score_right_grab_drawer
#             + self.score_object_in_drawer
#         ) / 4.0
#         self.current_tcp = tcp * 100.0

#         if self.current_tcp > self.max_tcp:
#             self.max_tcp = self.current_tcp

#     def get_TCP(self, use_max: bool = True) -> float:
#         """Return the task completion percentage at the end of an episode."""
#         tcp_value = self.max_tcp if use_max else self.current_tcp
#         self.reset_episode()
#         return tcp_value
