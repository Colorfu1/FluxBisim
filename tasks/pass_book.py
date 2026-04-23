import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402
from tasks.utils import check_gripper, generate_point_in_circle  # noqa: E402


class PassBook:
    BOOK_RANDOMIZATION_RADIUS = 0.02
    BOOK_RANDOMIZATION_RADIUS_NOISE = 0.04
    BOOKEND_RANDOMIZATION_RADIUS = 0.01
    BOOKEND_RANDOMIZATION_RADIUS_NOISE = 0.02

    def __init__(self):
        self.book_name = os.environ.get('BOOK_NAME', 'book')
        self.bookend_name = os.environ.get('BOOKEND_NAME', 'bookend')
        self.table_name = os.environ.get('TABLE_NAME', 'kitchen_table')

        self.objects = [
            BaseObject(self.book_name),
            BaseObject(self.bookend_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(book_name=self.book_name,
                               bookend_name=self.bookend_name)

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.book_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.BOOK_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))
            elif obj.name == self.bookend_name:
                default_position = generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.BOOKEND_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(
                    positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.book_name or obj.name == self.bookend_name:
                print(
                    'obj.name:',
                    obj.name,
                    'new position:',
                    obj.object.get_world_poses()[0].tolist()[0],
                )


class Checker:
    PLACE_THRESHOLD_MIN = 0.15  # Minimum distance to `book2` for the placement stage.
    PLACE_THRESHOLD_MAX = 0.2  # Maximum distance to `book2` for the placement stage.
    HEIGHT_MIN = 0.8  # Minimum valid placement height.
    HEIGHT_MAX = 0.88  # Maximum valid placement height.
    GRIPPER_THRESHOLD = 0.20  # Maximum gripper width for a valid grasp.

    def __init__(self, book_name, bookend_name):
        """Initialize the checker for the book placement task."""
        self.book_name = book_name
        self.bookend_name = bookend_name

    def check_gripper(self, env, hand='left'):
        """Check whether the gripper is holding the object."""
        return check_gripper(
            env,
            object_name=self.book_name,
            hand=hand,
            threshold=self.GRIPPER_THRESHOLD,
        )

    def check(self, env):
        """Check whether the book satisfies the placement conditions."""
        book = env.world.scene.get_object(self.book_name)
        bookend = env.world.scene.get_object(self.bookend_name)
        if book is None or bookend is None:
            raise ValueError('Book or bookend object does not exist.')

        book_pos = book.get_world_poses()[0][0]
        book_height = book_pos[2]

        bookend_pos = bookend.get_world_poses()[0][0]

        distance_xy = np.linalg.norm(book_pos[:2] - bookend_pos[:2])

        # The placement must satisfy both height and distance checks.
        height_check = self.HEIGHT_MIN <= book_height <= self.HEIGHT_MAX
        distance_check = (self.PLACE_THRESHOLD_MIN <= distance_xy <=
                          self.PLACE_THRESHOLD_MAX)

        return height_check and distance_check
