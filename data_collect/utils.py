"""
数据收集公共工具模块
提供HDF5文件操作、坐标转换等公共函数
"""

import os
from typing import Any, Dict

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


def relative_to_world_position(
    env, relative_position: np.ndarray, base_name: str = None
) -> np.ndarray:
    """
    将相对于机器人base的坐标转换为世界坐标

    Args:
        env: Isaac Sim环境实例
        relative_position: 相对于base的位置 [x, y, z]
        base_name: 机器人base名称，如果为None则从环境变量获取

    Returns:
        世界坐标位置 [x, y, z]
    """
    if base_name is None:
        base_name = os.environ.get("BASE_NAME", "base_kitchen")

    # 获取base的位置和姿态
    base_position = np.array([0.0, 0.0, 0.0])
    base_orientation = np.array([1.0, 0.0, 0.0, 0.0])

    if hasattr(env.robot, "robot_cfg") and base_name in env.robot.robot_cfg:
        base_cfg = env.robot.robot_cfg[base_name]
        base_position = np.array(base_cfg.get("position", [0.0, 0.0, 0.0]))
        base_orientation = np.array(base_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0]))

    # 将相对坐标转换为世界坐标
    # base_orientation格式: [w, x, y, z]
    base_rot = R.from_quat(
        base_orientation[[1, 2, 3, 0]]
    )  # 转换为scipy格式 [x, y, z, w]
    world_position = base_position + base_rot.apply(relative_position)

    return world_position


def init_hdf5(data_path: str, data: Dict[str, Any]) -> None:
    """
    初始化HDF5文件

    Args:
        data_path: HDF5文件路径
        data: 初始数据字典，包含图像和qpos等信息
    """
    with h5py.File(data_path, "w") as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        image = obs.create_group("images")

        dof_len = len(data["qpos"])

        for name, array in data.items():
            if "cam" in name:
                array = np.asarray(array)
                assert array.ndim == 3 and array.shape[-1] in (3, 4)
                h, w, c = array.shape
                _ = image.create_dataset(
                    name,
                    shape=(0, h, w, c),
                    maxshape=(None, h, w, c),
                    dtype="uint8",
                    chunks=(1, h, w, c),
                )

        _ = obs.create_dataset(
            "qpos",
            shape=(0, dof_len),
            maxshape=(None, dof_len),
            dtype=float,
            chunks=True,
        )
        _ = root.create_dataset(
            "action",
            shape=(0, dof_len),
            maxshape=(None, dof_len),
            dtype=float,
            chunks=True,
        )


def append_hdf5(data_path: str, data: Dict[str, Any]) -> None:
    """
    追加数据到HDF5文件

    Args:
        data_path: HDF5文件路径
        data: 数据字典，包含图像、qpos和action等信息
    """
    with h5py.File(data_path, "a") as root:
        images = root["observations/images"]
        obs = root["observations"]

        for name, value in data.items():
            if "cam" in name and isinstance(value, np.ndarray):
                images[name].resize(images[name].shape[0] + 1, axis=0)
                images[name][-1, ...] = value
            elif "qpos" in name:
                obs[name].resize(obs[name].shape[0] + 1, axis=0)
                obs[name][-1, ...] = value
            elif "action" in name:
                root[name].resize(root[name].shape[0] + 1, axis=0)
                root[name][-1, ...] = value


def extract_state_from_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    从观测数据中提取状态信息（相机图像和关节位置）

    Args:
        obs: 观测数据字典

    Returns:
        包含相机图像和qpos的状态字典
    """
    state = {name: value for name, value in obs.items() if "cam" in name}
    if "piper" in obs and "dof_pos" in obs["piper"]:
        q_pos = np.array(list(obs["piper"]["dof_pos"].values()), dtype=float)
        state["qpos"] = q_pos
    return state


def extract_action_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    从观测数据中提取动作（关节位置）

    Args:
        obs: 观测数据字典

    Returns:
        关节位置数组
    """
    if "piper" in obs and "dof_pos" in obs["piper"]:
        return np.array(list(obs["piper"]["dof_pos"].values()), dtype=float)
    return np.array([])
