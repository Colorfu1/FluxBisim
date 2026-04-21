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
        # 将相对路径转换为绝对路径
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
            # 使用safe_load避免安全风险
            config = yaml.safe_load(file)
        # return config[self.name.split("_")[0]]
        return config[self.name]

    def _resolve_usd_path(self, relative_path):
        """将相对路径转换为绝对路径"""
        # 获取object.yaml文件所在目录
        cfg_dir = os.path.dirname(__file__)
        base_dir = os.path.dirname(cfg_dir)

        # 构建绝对路径
        absolute_path = os.path.abspath(os.path.join(base_dir, relative_path))

        # 验证文件是否存在
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"USD文件不存在: {absolute_path}")

        return absolute_path

    def init_object(self, world, base_position=None, base_orientation=None):
        """将对象加入场景（XFormPrim 或 Articulation）

        Args:
            world: Isaac Sim 世界实例
            base_position: 机器人base的位置 [x, y, z]
            base_orientation: 机器人base的姿态 [w, x, y, z]
        """
        # print(f"正在初始化对象: {self.name}")
        if self.object is not None:
            raise ValueError(f"对象 '{self.name}' 已经被初始化")

        add_reference_to_stage(usd_path=self.usd_path, prim_path=f"/World/{self.name}")

        # 计算物体在世界坐标系中的位置和姿态
        if base_position is not None and base_orientation is not None:
            # 物体相对于base的位置和姿态
            obj_relative_pos = np.array(self.init_position)
            obj_relative_quat = np.array(self.init_orientation)

            # 将相对位姿变换到世界坐标系
            base_rot = R.from_quat(
                base_orientation[[1, 2, 3, 0]]
            )  # 转换为scipy格式 [x,y,z,w]

            obj_world_pos = base_position + base_rot.apply(obj_relative_pos)
            self.init_poisition_global = obj_world_pos
            obj_world_quat = (
                base_rot * R.from_quat(obj_relative_quat[[1, 2, 3, 0]])
            ).as_quat()[[3, 0, 1, 2]]  # 转换回[w,x,y,z]格式

            positions = np.array([obj_world_pos])
            orientations = np.array([obj_world_quat])
        else:
            # 如果没有提供base位姿，使用原始位置（假设已经是世界坐标系）
            self.init_poisition_global = self.init_position
            positions = np.array([self.init_position])
            orientations = np.array([self.init_orientation])

        self.object = XFormPrim(
            prim_paths_expr=f"/World/{self.name}",
            name=self.name,
            # usd_path=self.usd_path,
            positions=positions,
            orientations=orientations,
        )
        world.scene.add(self.object)

    def get_obj_pos(self):
        # print(obj.get_world_poses()[0].tolist()[0])
        return {
            "pos": self.object.get_world_poses()[0].tolist()[0],
            "rot": self.object.get_world_poses()[1].tolist()[0],
        }
        # 设置其他属性
        # self.object.set_name(self.name)
