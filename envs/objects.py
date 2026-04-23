import os

import numpy as np
import yaml
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from scipy.spatial.transform import Rotation as R


class BaseObject:
    def __init__(self, name, init_position=None, init_orientation=None, usd_path=None):
        self.name = name
        self.cfg = self._load_object_config()
        self.usd_path = self._resolve_usd_path(self.cfg["usd_path"])
        self.init_position = (
            self.cfg["position"] if init_position is None else init_position
        )
        self.init_orientation = (
            self.cfg["orientation"] if init_orientation is None else init_orientation
        )
        self.init_poisition_global = None
        self.object = None

    def _load_object_config(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "cfg", "objects.yaml")
        with open(cfg_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config[self.name]

    def _resolve_usd_path(self, relative_path):
        """Resolve a USD path from the config into an absolute path."""
        cfg_dir = os.path.dirname(__file__)
        base_dir = os.path.dirname(cfg_dir)

        absolute_path = os.path.abspath(os.path.join(base_dir, relative_path))

        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"USD file does not exist: {absolute_path}")

        return absolute_path

    def init_object(self, world, base_position=None, base_orientation=None):
        """Add the object to the scene as an ``XFormPrim`` (or Articulation).

        Args:
            world: The Isaac Sim world instance.
            base_position: Robot base position ``[x, y, z]``.
            base_orientation: Robot base orientation quaternion ``[w, x, y, z]``.
        """
        if self.object is not None:
            raise ValueError(f"Object '{self.name}' has already been initialized.")

        add_reference_to_stage(usd_path=self.usd_path, prim_path=f"/World/{self.name}")

        # Compute the object pose in the world frame.
        if base_position is not None and base_orientation is not None:
            # Object pose relative to the robot base.
            obj_relative_pos = np.array(self.init_position)
            obj_relative_quat = np.array(self.init_orientation)

            # Transform the relative pose into the world frame.
            base_rot = R.from_quat(
                base_orientation[[1, 2, 3, 0]]
            )  # Convert [w,x,y,z] to scipy order [x,y,z,w].

            obj_world_pos = base_position + base_rot.apply(obj_relative_pos)
            self.init_poisition_global = obj_world_pos
            obj_world_quat = (
                base_rot * R.from_quat(obj_relative_quat[[1, 2, 3, 0]])
            ).as_quat()[[3, 0, 1, 2]]  # Convert back to [w,x,y,z] order.

            positions = np.array([obj_world_pos])
            orientations = np.array([obj_world_quat])
        else:
            # When no base pose is provided, treat the configured pose as world-frame.
            self.init_poisition_global = self.init_position
            positions = np.array([self.init_position])
            orientations = np.array([self.init_orientation])

        self.object = XFormPrim(
            prim_paths_expr=f"/World/{self.name}",
            name=self.name,
            positions=positions,
            orientations=orientations,
        )
        world.scene.add(self.object)

    def get_obj_pos(self):
        return {
            "pos": self.object.get_world_poses()[0].tolist()[0],
            "rot": self.object.get_world_poses()[1].tolist()[0],
        }
