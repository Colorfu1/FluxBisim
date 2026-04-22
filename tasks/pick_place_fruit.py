import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import check_gripper, generate_point_in_circle  # noqa: E402


class PickPlaceFruit:
    RANDOMIZATION_RADIUS = 0.04
    RANDOMIZATION_RADIUS_NOISE = 0.06

    def __init__(self):
        self.fruit_name = os.environ.get("FRUIT_NAME", "banana")
        self.plate_name = os.environ.get("PLATE_NAME", "plate")
        self.table_name = os.environ.get("TABLE_NAME", "kitchen_table")

        self.objects = [BaseObject(self.fruit_name), BaseObject(self.plate_name)]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(self.fruit_name, self.plate_name)

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.fruit_name or obj.name == self.plate_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(positions=np.array([default_position]))

            # Apply the new default state before moving the object.
            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.fruit_name or obj.name == self.plate_name:
                print(
                    "obj.name:",
                    obj.name,
                    "new position:",
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    PLACE_THRESHOLD_MIN = 0.06
    GRIPPER_THRESHOLD = 0.15

    def __init__(self, fruit_name, plate_name):
        self.fruit_name = fruit_name
        self.plate_name = plate_name

    def check_gripper(self, env):
        """Check whether the gripper is holding the object."""
        return check_gripper(
            env,
            object_name=self.fruit_name,
            hand="right",
            threshold=self.GRIPPER_THRESHOLD,
        )

    def check(self, env):
        """Check whether the object is at the target position."""
        fruit = env.world.scene.get_object(self.fruit_name)
        plate = env.world.scene.get_object(self.plate_name)
        if fruit is None or plate is None:
            raise ValueError("Fruit or plate object does not exist.")

        fruit_pos = fruit.get_world_poses()[0][0]
        plate_pos = plate.get_world_poses()[0][0]

        distance_xy = np.linalg.norm(fruit_pos[:2] - plate_pos[:2])

        return (
            distance_xy <= self.PLACE_THRESHOLD_MIN
            and not self.check_gripper(env)
        )  # The position threshold can be tuned if needed.
