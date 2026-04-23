import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import check_gripper  # noqa: E402
from tasks.utils import generate_point_in_circle  # noqa: E402
from tasks.utils import get_prim_world_pos  # noqa: E402


class PullPushDrawer:
    RANDOMIZATION_RADIUS = 0.01
    RANDOMIZATION_RADIUS_NOISE = 0.02

    def __init__(self):
        self.object_name = os.environ.get('OBJECT_NAME', 'apple_drawer')
        self.drawer_name = os.environ.get('DRAWER_NAME', 'drawer')
        self.table_name = os.environ.get('TABLE_NAME', 'kitchen_table')

        self.objects = [
            BaseObject(self.object_name),
            BaseObject(self.drawer_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(object_name=self.object_name,
                               drawer_name=self.drawer_name)

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.object_name or obj.name == self.drawer_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.object_name or obj.name == self.drawer_name:
                print(
                    'obj.name:',
                    obj.name,
                    'new position:',
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
        return check_gripper(
            env,
            object_name=self.object_name,
            hand='left',
            threshold=self.GRIPPER_THRESHOLD,
        )

    def _get_drawer_distance(self, env):
        """Calculate the current drawer opening distance."""
        drawer_front_pos = get_prim_world_pos(env.world,
                                              '/World/drawer/E_drawer2_7')
        target_part_pos = get_prim_world_pos(
            env.world, '/World/drawer/E_body_1/E_part4_5')
        return np.linalg.norm(drawer_front_pos -
                              target_part_pos) - self.DRAWER_OFFSET

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
            raise ValueError('Object or drawer does not exist.')

        object_pos = object.get_world_poses()[0][0]
        drawer_pos = get_prim_world_pos(
            env.world, '/World/drawer/E_drawer2_7') + np.array(
                [-0.063, 0.083, 0.0018])

        distance_xy = np.linalg.norm(object_pos[:2] - drawer_pos[:2])

        return distance_xy <= self.POSITION_THRESHOLD and self.check_drawer_closed(
            env)
