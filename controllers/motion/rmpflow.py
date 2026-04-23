import os

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation


class RMPFlowController(mg.MotionPolicyController):

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 30.0,
    ) -> None:
        motion_path = os.path.dirname(__file__)
        cfg_path = os.path.join(motion_path, 'rmpflow')
        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(cfg_path,
                                                'piper_description.yaml'),
            rmpflow_config_path=os.path.join(cfg_path,
                                             'piper_rmpflow_config.yaml'),
            urdf_path=os.path.join(motion_path, 'piper_description.urdf'),
            end_effector_frame_name='gripper_base',
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose(
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
