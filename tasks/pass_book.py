import os

import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from envs import BaseObject  # noqa: E402


class PassBook:
    BOOK_RANDOMIZATION_RADIUS = 0.02
    BOOK_RANDOMIZATION_RADIUS_NOISE = 0.04
    BOOKEND_RANDOMIZATION_RADIUS = 0.01
    BOOKEND_RANDOMIZATION_RADIUS_NOISE = 0.02

    def __init__(self):
        self.book_name = os.environ.get("BOOK_NAME", "book")
        self.bookend_name = os.environ.get("BOOKEND_NAME", "bookend")
        self.table_name = os.environ.get("TABLE_NAME", "kitchen_table")

        self.objects = [
            BaseObject(self.book_name),
            BaseObject(self.bookend_name),
        ]
        self.environment = BaseObject(self.table_name)
        self.checker = Checker(book_name=self.book_name, bookend_name=self.bookend_name)

    def _generate_point_in_circle(self, origin_x, origin_y, origin_z, radius=0.02):
        """Sample a random point inside a circle."""
        theta = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        x = origin_x + r * np.cos(theta)
        y = origin_y + r * np.sin(theta)
        return [x, y, origin_z]

    def randomize_objects_position(self):
        """Randomize object positions around their initial poses."""
        for obj in self.objects:
            if obj.name == self.book_name:
                default_position = self._generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.BOOK_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(positions=np.array([default_position]))
            elif obj.name == self.bookend_name:
                default_position = self._generate_point_in_circle(
                    obj.init_poisition_global[0],
                    obj.init_poisition_global[1],
                    obj.init_poisition_global[2],
                    radius=self.BOOKEND_RANDOMIZATION_RADIUS,
                )
                obj.object.set_default_state(positions=np.array([default_position]))

            obj.object.post_reset()

        for obj in self.objects:
            if obj.name == self.book_name or obj.name == self.bookend_name:
                print(
                    "obj.name:",
                    obj.name,
                    "new position:",
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

    def check_gripper(self, env, hand="left"):
        """Check whether the gripper is holding the object."""
        manipulator = (
            env.robot.left_manipulator
            if hand == "left"
            else env.robot.right_manipulator
        )
        current_ee_pos = manipulator.end_effector.get_world_pose()[0] + np.array(
            [0.0, 0.0, -0.13]
        )

        book_pos = env.world.scene.get_object(self.book_name).get_world_poses()[0][0]

        # Measure the distance between the end effector and the object.
        ee_distance = np.linalg.norm(current_ee_pos - book_pos)

        if ee_distance > self.GRIPPER_THRESHOLD:
            print("The gripper is not holding the object.")
            return False
        else:
            return True

    def check(self, env):
        """Check whether the book satisfies the placement conditions."""
        book = env.world.scene.get_object(self.book_name)
        bookend = env.world.scene.get_object(self.bookend_name)
        if book is None or bookend is None:
            raise ValueError("Book or bookend object does not exist.")

        book_pos = book.get_world_poses()[0][0]
        book_height = book_pos[2]

        bookend_pos = bookend.get_world_poses()[0][0]

        distance_xy = np.linalg.norm(book_pos[:2] - bookend_pos[:2])

        # The placement must satisfy both height and distance checks.
        height_check = self.HEIGHT_MIN <= book_height <= self.HEIGHT_MAX
        distance_check = (
            self.PLACE_THRESHOLD_MIN <= distance_xy <= self.PLACE_THRESHOLD_MAX
        )

        return height_check and distance_check


# class Checker:
#     """Task completion checker."""
#     LEFT_HAND_THRESHOLD = 0.3  # The book must leave its initial position by at least 0.3 m.
#     PLACE_THRESHOLD_MIN = 0.15  # Minimum distance to `book2` for the placement stage.
#     PLACE_THRESHOLD_MAX = 0.2  # Maximum distance to `book2` for the placement stage.
#     HEIGHT_MIN = 0.8  # Minimum valid placement height.
#     HEIGHT_MAX = 0.88  # Maximum valid placement height.

#     def __init__(self, book_name, bookend_name):
#         """Initialize the checker for the book placement task."""
#         self.book_name = book_name
#         self.bookend_name = bookend_name
#         self.book_initial_position = None

#     def _get_book_initial_position(self, env):
#         """Return the initial book position and refresh it after a reset."""
#         book = env.world.scene.get_object(self.book_name)
#         current_pos = book.get_world_poses()[0][0]

#         # Cache the first observed position after reset.
#         if self.book_initial_position is None:
#             self.book_initial_position = current_pos.copy()
#         else:
#             # Refresh the cached value when a reset likely happened.
#             distance = np.linalg.norm(current_pos - self.book_initial_position)
#             if distance > 1.0:
#                 self.book_initial_position = current_pos.copy()

#         return self.book_initial_position

#     def check(self, env):
#         """Check whether the book satisfies the placement conditions."""
#         book = env.world.scene.get_object(self.book_name)
#         bookend = env.world.scene.get_object(self.bookend_name)
#         if book is None or bookend is None:
#             raise ValueError(f"Book or bookend object does not exist.")

#         book_pos = book.get_world_poses()[0][0]
#         book_height = book_pos[2]

#         bookend_pos = bookend.get_world_poses()[0][0]

#         distance_xy = np.linalg.norm(book_pos[:2] - bookend_pos[:2])

#         # The placement must satisfy both height and distance checks.
#         height_check = (self.HEIGHT_MIN <= book_height <= self.HEIGHT_MAX)
#         distance_check = (self.PLACE_THRESHOLD_MIN <= distance_xy <= self.PLACE_THRESHOLD_MAX)

# # Print detailed diagnostics for data collection.
# print("=" * 60)
# print("[Data Collection Check]")
# print("  Condition 1 - Height check:")
# print(f"    Actual height: {book_height:.4f} m")
# print(f"    Valid range: {self.HEIGHT_MIN:.2f} m to {self.HEIGHT_MAX:.2f} m")
# print(f"    Result: {'PASS' if height_check else 'FAIL'}")
# print("  Condition 2 - Distance check:")
# print(f"    Actual distance: {distance_xy:.4f} m")
# print(
#     f"    Valid range: {self.PLACE_THRESHOLD_MIN:.4f} m to "
#     f"{self.PLACE_THRESHOLD_MAX:.4f} m"
# )
# print(f"    Result: {'PASS' if distance_check else 'FAIL'}")
# print(
#     f"  Final result: {'PASS - Save data' if (height_check and distance_check) else 'FAIL - Discard data'}"
# )
# print("=" * 60)

# return height_check and distance_check

# def check_eval(self, env, mode="SR"):
#     """Evaluate task completion for success rate or TCP."""
#     book = env.world.scene.get_object(self.book_name)
#     bookend = env.world.scene.get_object(self.bookend_name)

#     # Get the current book pose.
#     book_pos = book.get_world_poses()[0][0]
#     book_height = book_pos[2]

#     bookend_pos = bookend.get_world_poses()[0][0]

#     # Get the initial pose recorded after reset.
#     book_initial_pos = self._get_book_initial_position(env)

#     # Get the target shelf pose.
#     book2_pos = book2.get_world_poses()[0][0]

#     # Measure how far the book moved away from its initial pose.
#     distance_from_initial_xy = np.linalg.norm(obj1_pos[:2] - book_initial_pos[:2])

#     # Measure the planar distance from the book to the target shelf.
#     distance_to_book2_xy = np.linalg.norm(obj1_pos[:2] - book2_pos[:2])

#     # The task is split into a handoff stage and a placement stage.
#     left_hand_check = distance_from_initial_xy > self.LEFT_HAND_THRESHOLD
#     height_check = (self.HEIGHT_MIN <= obj1_height <= self.HEIGHT_MAX)
#     distance_check = (self.PLACE_THRESHOLD_MIN <= distance_to_book2_xy <= self.PLACE_THRESHOLD_MAX)
#     right_hand_check = height_check and distance_check

#     if mode == "SR":
#         return left_hand_check and right_hand_check
#     elif mode == "TCP":
#         left_hand_score = 0.5 if left_hand_check else 0.0
#         right_hand_score = 0.5 if right_hand_check else 0.0
#         tcp = left_hand_score + right_hand_score
#         return tcp
#     else:
#         return left_hand_check and right_hand_check

# def get_tcp_details(self, env):
#     """Return a detailed TCP breakdown for debugging."""
#     obj1 = env.world.scene.get_object(self.obj_name1)
#     book2 = env.world.scene.get_object("book2")

#     # Get the current book pose.
#     obj1_pos = obj1.get_world_poses()[0][0]
#     obj1_height = obj1_pos[2]

#     # Get the initial pose recorded after reset.
#     book_initial_pos = self._get_book_initial_position(env)

#     # Get the target shelf pose.
#     book2_pos = book2.get_world_poses()[0][0]

#     # Measure how far the book moved away from its initial pose.
#     distance_from_initial_xy = np.linalg.norm(obj1_pos[:2] - book_initial_pos[:2])

#     # Measure the planar distance from the book to the target shelf.
#     distance_to_book2_xy = np.linalg.norm(obj1_pos[:2] - book2_pos[:2])

#     # Stage 1: the book should move far enough from the initial pose.
#     left_hand_check = distance_from_initial_xy > self.LEFT_HAND_THRESHOLD
#     left_hand_diff = distance_from_initial_xy - self.LEFT_HAND_THRESHOLD
#     left_hand_status = "pass" if left_hand_check else "fail"

#     # Stage 2: the placement must satisfy both height and distance checks.
#     height_check = (self.HEIGHT_MIN <= obj1_height <= self.HEIGHT_MAX)
#     distance_check = (self.PLACE_THRESHOLD_MIN <= distance_to_book2_xy <= self.PLACE_THRESHOLD_MAX)
#     right_hand_check = height_check and distance_check
#     right_hand_status = "pass" if right_hand_check else "fail"

#     return {
#         "left_hand": {
#             "status": left_hand_status,
#             "diff": left_hand_diff,
#             "distance": distance_from_initial_xy,
#             "threshold": self.LEFT_HAND_THRESHOLD,
#             "initial_position": book_initial_pos.tolist(),
#             "current_position": obj1_pos.tolist()
#         },
#         "right_hand": {
#             "status": right_hand_status,
#             "height_check": height_check,
#             "distance_check": distance_check,
#             "height": obj1_height,
#             "height_min": self.HEIGHT_MIN,
#             "height_max": self.HEIGHT_MAX,
#             "distance": distance_to_book2_xy,
#             "threshold_min": self.PLACE_THRESHOLD_MIN,
#             "threshold_max": self.PLACE_THRESHOLD_MAX
#         },
#         "book_data": {
#             "current_position": obj1_pos.tolist(),
#             "initial_position": book_initial_pos.tolist(),
#             "book2_position": book2_pos.tolist()
#         }
#     }
