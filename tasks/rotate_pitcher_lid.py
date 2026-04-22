import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import (  # noqa: E402
    check_gripper,
    generate_point_in_circle,
    get_prim_world_pos,
)


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

    def randomize_objects_position(self):
        """Randomize the pitcher and lid around their initial poses."""
        for obj in self.objects:
            if obj.name == self.pitcher_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.PITCHER_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(positions=np.array([default_position]))
            elif obj.name == self.lid_name:
                default_position = generate_point_in_circle(
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
        return check_gripper(
            env,
            object_name=self.lid_name,
            hand="right",
            threshold=self.GRIPPER_THRESHOLD,
        )

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
