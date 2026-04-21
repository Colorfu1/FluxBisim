import typing

import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators import controllers as manipulators_controllers
from isaacsim.robot.manipulators.grippers import ParallelGripper

from controllers.motion import RMPFlowController


def _mix_sin(t):
    return 0.5 * (1 - np.cos(t * np.pi))


def _combine_convex(a, b, alpha):
    return (1 - alpha) * a + alpha * b


class LeftArmPickPlaceController(manipulators_controllers.PickPlaceController):
    """
    A simple pick and place state machine for tutorials

    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt=None,
        end_effector_initial_height: float = 1.2,
    ) -> None:
        if events_dt is None:
            events_dt = [0.01, 0.01, 0.5, 0.2, 0.008, 0.004, 0.008, 0.4, 0.01, 0.01]

        super().__init__(
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=end_effector_initial_height,
        )

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 0
        else:
            raise ValueError()


class RightArmPullPushController(BaseController):
    """
    A pull and push state machine for the right arm.

    Each phase runs for 1 second, which is the internal time of the state machine.
    Dt of each phase/event step is defined separately.

    - Phase 0: Move end_effector above the drawer handle at the initial height.
    - Phase 1: Lower end_effector down to the drawer handle height.
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: Close gripper to grasp the handle.
    - Phase 4: Pull the drawer.
    - Phase 5: Wait for Robot's inertia to settle after pulling.
    - Phase 6: Push the drawer.
    - Phase 7: Open gripper to release the handle.
    - Phase 8: Move end_effector back to the initial height.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        gripper: ParallelGripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008]

        cspace_controller = RMPFlowController(
            name=name + "_cspace_controller", robot_articulation=robot_articulation
        )

        super().__init__(name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.9
        self._h0 = None
        self._events_dt = events_dt

        if not isinstance(self._events_dt, np.ndarray) and not isinstance(
            self._events_dt, list
        ):
            raise Exception("events dt need to be list or numpy array")
        elif isinstance(self._events_dt, np.ndarray):
            self._events_dt = self._events_dt.tolist()
        if len(self._events_dt) > 9:
            raise Exception("events dt length must be less than or equal to 9")

        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        return

    def is_paused(self) -> bool:
        return self._pause

    def get_current_event(self) -> int:
        return self._event

    def _get_alpha(self):
        if self._event < 4:
            return 0
        elif self._event in [4, 6]:
            return _mix_sin(self._t)
        elif self._event == 8:
            return 1.0
        else:
            raise ValueError()

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        if self._event == 6:
            start = np.array([target_x, target_y])
            end = np.array([current_x, current_y])
        else:
            start = np.array([current_x, current_y])
            end = np.array([target_x, target_y])
        return _combine_convex(start, end, alpha)

    def _get_target_hs(self):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            h = _combine_convex(self._h1, self._h0, _mix_sin(self._t))
        elif self._event in [4, 6]:
            h = self._h0
        elif self._event == 8:
            h = _combine_convex(self._h0, self._h1, _mix_sin(self._t))
        else:
            raise ValueError()
        return h

    def forward(
        self,
        push_position: np.ndarray,
        pull_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])

        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event in [2, 5]:
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="open")
        else:
            if self._event in [0, 1]:
                self._current_target_x = push_position[0]
                self._current_target_y = push_position[1]
                self._h0 = push_position[2]

            interpolated_xy = self._get_interpolated_xy(
                pull_position[0],
                pull_position[1],
                self._current_target_x,
                self._current_target_y,
            )
            target_height = self._get_target_hs()
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            orientation_target = end_effector_orientation

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=orientation_target,
            )

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joint_positions

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(
                self._events_dt, list
            ):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 9:
                raise Exception("events dt length must be less than or equal to 9")
        return

    def is_done(self) -> bool:
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        self._pause = True
        return

    def resume(self) -> None:
        self._pause = False
        return
