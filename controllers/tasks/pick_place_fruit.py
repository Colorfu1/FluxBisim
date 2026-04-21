import typing

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators import controllers as manipulators_controllers
from isaacsim.robot.manipulators.grippers import ParallelGripper

from controllers.motion import RMPFlowController


class PickPlaceFruitController(manipulators_controllers.PickPlaceController):
    """
    Pick and place controller without the final retreat-to-old-xy phase.

    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt=None,
        end_effector_initial_height: float = 0.9,
    ) -> None:
        if events_dt is None:
            events_dt = [0.008, 0.008, 0.5, 0.2, 0.008, 0.004, 0.008, 0.4, 0.008]

        if len(events_dt) != 9:
            raise Exception("events dt length must be 9")

        super().__init__(
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=end_effector_initial_height,
        )

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        if events_dt is not None:
            if not isinstance(events_dt, np.ndarray) and not isinstance(
                events_dt, list
            ):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(events_dt, np.ndarray):
                events_dt = events_dt.tolist()
            if len(events_dt) != 9:
                raise Exception("events dt length must be 9")

        super().reset(
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )
