import os

import numpy as np
import rootutils
from isaacsim.core.prims import XFormPrim
from scipy.spatial.transform import Rotation as R

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402


def get_prim_world_quat(world, prim_path: str):
    """Return the world position `(x, y, z)` of a prim."""
    prim_view = XFormPrim(
        prim_paths_expr=prim_path,
        name=f"tmp_{prim_path.split('/')[-1]}",
    )
    _, quat = prim_view.get_world_poses()
    return np.array(quat[0])


class CloseBox:
    NUT_RANDOMIZATION_RADIUS = 0.03
    NUT_RANDOMIZATION_RADIUS_NOISE = 0.06

    def __init__(self):
        self.nut_name = os.environ.get("NUT_NAME", "nut")
        self.box_name = os.environ.get("BOX_NAME", "box")
        self.table_name = os.environ.get("TABLE_NAME", "industry_table")

        self.objects = [
            BaseObject(self.nut_name),
            BaseObject(self.box_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(nut_name=self.nut_name, box_name=self.box_name)

    def _generate_point_in_circle(self, origin_x, origin_y, origin_z, radius=0.03):
        """Sample a random point inside a circle."""
        theta = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        x = origin_x + r * np.cos(theta)
        y = origin_y + r * np.sin(theta)
        return [x, y, origin_z]

    def randomize_objects_position(self):
        """Randomize only the nut while keeping the box fixed."""
        for obj in self.objects:
            if obj.name == self.nut_name:
                default_position = self._generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.NUT_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.nut_name or obj.name == self.box_name:
                print(
                    "obj.name:",
                    obj.name,
                    "new position:",
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    PLACE_THRESHOLD = 0.15
    CLOSE_THRESHOLD_MIN_DEG = 10.0
    CLOSE_THRESHOLD_MAX_DEG = 15.0
    GRIPPER_THRESHOLD = 0.2

    def __init__(self, nut_name, box_name):
        self.nut_name = nut_name
        self.box_name = box_name

    def check_gripper(self, env):
        """Check whether the left gripper is holding the nut."""
        current_ee_pos = env.robot.left_manipulator.end_effector.get_world_pose()[
            0
        ] + np.array([0.0, 0.0, -0.13])
        nut_pos = np.array(
            env.world.scene.get_object(self.nut_name).get_world_poses()[0][0]
        )

        ee_distance = np.linalg.norm(current_ee_pos - nut_pos)

        if ee_distance > self.GRIPPER_THRESHOLD:
            print("The gripper is not holding the object.")
            return False
        else:
            return True

    def check(self, env):
        """Check whether both the nut is in the box and the box is closed."""
        nut = env.world.scene.get_object(self.nut_name)
        box = env.world.scene.get_object(self.box_name)
        if nut is None or box is None:
            raise ValueError("Nut or box object does not exist.")

        nut_pos = np.array(nut.get_world_poses()[0][0])
        box_pos = np.array(box.get_world_poses()[0][0])
        box_lid_quat = get_prim_world_quat(env.world, f"/World/{self.box_name}/box_lid")
        box_lid_euler = R.from_quat(
            [box_lid_quat[1], box_lid_quat[2], box_lid_quat[3], box_lid_quat[0]]
        ).as_euler(
            "xyz",
            degrees=True,
        )

        distance_xy = np.linalg.norm(nut_pos[:2] - box_pos[:2])

        return (
            distance_xy <= self.PLACE_THRESHOLD
            and self.CLOSE_THRESHOLD_MIN_DEG
            <= box_lid_euler[1]
            <= self.CLOSE_THRESHOLD_MAX_DEG
        )
