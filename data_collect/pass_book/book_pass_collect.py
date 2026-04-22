from omni.isaac.kit import SimulationApp

sim_app = SimulationApp({"headless": False})

import argparse  # noqa: E402
import os  # noqa: E402
from typing import Any, Dict, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import rootutils  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

rootutils.setup_root(__file__, pythonpath=True)

from controllers.tasks.pass_book import (  # noqa: E402
    LeftArmPickPlaceController,
    RightArmPickPlaceController,
)
from data_collect.utils import (  # noqa: E402
    append_hdf5,
    extract_action_from_obs,
    extract_state_from_obs,
    init_hdf5,
    relative_to_world_position,
)
from envs import IsaacsimEnv  # noqa: E402


def setup_controllers(
    env: IsaacsimEnv,
) -> Tuple[LeftArmPickPlaceController, RightArmPickPlaceController]:
    """Set up controllers for the left and right arms."""
    left_events_dt = [0.01, 0.01, 0.5, 0.05, 0.05, 0.01]
    right_events_dt = [0.008, 0.008, 0.5, 0.03, 0.01, 0.01, 0.01, 0.01, 0.05, 0.02]

    left_controller = LeftArmPickPlaceController(
        name="left_controller",
        robot_articulation=env.robot.left_manipulator,
        gripper=env.robot.left_manipulator.gripper,
        events_dt=left_events_dt,
    )

    right_controller = RightArmPickPlaceController(
        name="right_controller",
        robot_articulation=env.robot.right_manipulator,
        gripper=env.robot.right_manipulator.gripper,
        events_dt=right_events_dt,
    )
    return left_controller, right_controller


def compute_left_arm_move_action(
    env: IsaacsimEnv,
    left_controller: LeftArmPickPlaceController,
    move_state: Dict[str, Any],
    target_orientation: np.ndarray,
) -> ArticulationAction:
    """Compute the action for moving the left arm to a target position."""
    target_position = move_state["target_position"]

    if move_state["progress"] < 1.0:
        if move_state["start_position"] is None:
            move_state["start_position"] = np.array(
                env.robot.left_manipulator.end_effector.get_world_pose()[0]
            )

        move_alpha = 0.5 * (1 - np.cos(move_state["progress"] * np.pi))
        target_position = move_state["start_position"] + move_alpha * (
            move_state["target_position"] - move_state["start_position"]
        )
        move_state["progress"] = min(1.0, move_state["progress"] + move_state["dt"])

    position_action = left_controller._cspace_controller.forward(
        target_end_effector_position=target_position,
        target_end_effector_orientation=target_orientation,
    )
    arm_joint_positions = np.asarray(position_action.joint_positions)[:6]
    gripper_joint_positions = left_controller._gripper.forward(
        action="open"
    ).joint_positions[-2:]

    return ArticulationAction(
        joint_positions=list(arm_joint_positions) + list(gripper_joint_positions)
    )


def collect_episode_data(
    env: IsaacsimEnv,
    left_controller: LeftArmPickPlaceController,
    right_controller: RightArmPickPlaceController,
    book_name: str,
    bookend_name: str,
) -> Dict[str, Any]:
    """Collect data for a single episode with dual-arm collaboration."""
    obs = env.step()

    episode_data = {"states": [], "actions": []}

    cnt = 0
    sample_step = 1

    picking_position_1 = np.array(obs[book_name]["pos"]) + np.array([-0.01, 0.0, -0.14])
    handover_position_1 = relative_to_world_position(env, np.array([0.45, 0.1, 0.4]))
    picking_orientation_1 = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))
    handover_orientation_1 = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))

    # Phase 1: the left arm performs the pick action
    while not left_controller.is_done():
        event = left_controller.get_current_event()

        if event in [3, 4, 5]:
            if not env.task.checker.check_gripper(env, hand="left"):
                print("The gripper is not holding the object.")
                break

        if cnt % sample_step == 0:
            state = extract_state_from_obs(obs)
            episode_data["states"].append(state)

        left_action_1 = left_controller.forward(
            picking_position=picking_position_1,
            placing_position=handover_position_1,
            current_joint_positions=np.array(
                list(obs["piper"]["dof_pos"].values())[:8], dtype=float
            ),
            end_effector_offset=np.array([-0.03, 0.0, 0.28]),
            picking_orientation=picking_orientation_1,
            placing_orientation=handover_orientation_1,
        )

        obs = env.step([left_action_1, None])

        if event in [3, 4, 5]:
            obs["piper"]["dof_pos"]["llink7"] = 0.0
            obs["piper"]["dof_pos"]["llink8"] = 0.0

        if cnt % sample_step == 0:
            action = extract_action_from_obs(obs)
            episode_data["actions"].append(action)

        cnt += 1

    last_left_action = left_action_1

    left_move_state = {
        "started": False,
        "progress": 0.0,
        "dt": 0.008,
        "start_position": None,
        "target_position": None,
    }

    # Phase 2: the right arm performs pick-and-place while the left arm releases the object and moves away after the right arm grasps it
    last_event = -1

    picking_position_2 = np.array(obs[book_name]["pos"]) + np.array(
        [0.282, 0.125, -0.179]
    )
    placing_position_2 = np.array(obs[bookend_name]["pos"]) + np.array(
        [-0.08, 0.05, 0.11]
    )
    picking_orientation_2 = euler_angles_to_quat(np.array([0.0, 3 * np.pi / 4, np.pi]))
    placing_orientation_2 = euler_angles_to_quat(
        np.array([0.0, 3 * np.pi / 4, np.pi / 2])
    )

    move_position = np.array(obs[book_name]["pos"]) + np.array([-0.5, 0.1, -0.165])
    _, move_orientation = env.robot.left_manipulator.end_effector.get_world_pose()

    while not right_controller.is_done():
        event = right_controller.get_current_event()

        if event in [3, 4, 5, 6, 7]:
            if not env.task.checker.check_gripper(env, hand="right"):
                print("The gripper is not holding the object.")
                break

        if cnt % sample_step == 0:
            state = extract_state_from_obs(obs)
            episode_data["states"].append(state)

        if last_event == 3 and event != last_event:
            current_left_position, _ = (
                env.robot.left_manipulator.end_effector.get_world_pose()
            )
            left_move_state["started"] = True
            left_move_state["progress"] = 0.0
            left_move_state["start_position"] = np.array(current_left_position)
            left_move_state["target_position"] = move_position

        if event != last_event:
            last_event = event

        right_actions_2 = right_controller.forward(
            picking_position=picking_position_2,
            placing_position=placing_position_2,
            current_joint_positions=np.array(
                list(obs["piper"]["dof_pos"].values())[8:], dtype=float
            ),
            end_effector_offset=np.array([-0.1, 0, 0.45]),
            picking_orientation=picking_orientation_2,
            placing_orientation=placing_orientation_2,
        )

        if left_move_state["started"]:
            left_actions_2 = compute_left_arm_move_action(
                env=env,
                left_controller=left_controller,
                move_state=left_move_state,
                target_orientation=move_orientation,
            )

            if left_move_state["progress"] >= 1.0:
                last_left_action = left_actions_2
                left_move_state["started"] = False
        else:
            left_actions_2 = last_left_action

        obs = env.step([left_actions_2, right_actions_2])

        if event in [0, 1, 2, 3]:
            obs["piper"]["dof_pos"]["llink7"] = 0.0
            obs["piper"]["dof_pos"]["llink8"] = 0.0
        if event in [3, 4, 5, 6, 7]:
            obs["piper"]["dof_pos"]["rlink7"] = 0.0
            obs["piper"]["dof_pos"]["rlink8"] = 0.0

        if cnt % sample_step == 0:
            action = extract_action_from_obs(obs)
            episode_data["actions"].append(action)

        cnt += 1

    return episode_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="apartment")
    parser.add_argument("--config", type=str, default="book_pass_config.yaml")
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to save the collected data"
    )
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(cfg_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    table_name = f"{args.env}_table"
    os.environ["TABLE_NAME"] = table_name
    base_name = f"base_{args.env}"
    os.environ["BASE_NAME"] = base_name

    book_name = config.get("book", "book")
    os.environ["BOOK_NAME"] = book_name
    bookend_name = config.get("bookend", "bookend")
    os.environ["BOOKEND_NAME"] = bookend_name

    env = IsaacsimEnv(config)
    env.reset()
    max_demo = config.get("max_demo", 100)
    episodes = []

    for i in range(60):
        env.world.step(render=True)

    while sim_app.is_running() and len(episodes) < max_demo:
        print(f"Current episode: {len(episodes) + 1}/{max_demo}")

        env.reset()
        for i in range(60):
            env.world.step(render=True)

        left_controller, right_controller = setup_controllers(env)

        for i in range(60):
            env.world.step(render=True)

        episode_data = collect_episode_data(
            env,
            left_controller,
            right_controller,
            book_name=book_name,
            bookend_name=bookend_name,
        )

        if env.task.checker.check(env):
            episodes.append(True)

            if args.data_path:
                data_path = args.data_path
            else:
                data_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "data",
                    args.env,
                    env.task_name,
                )

            os.makedirs(data_path, exist_ok=True)
            h5_path = os.path.join(data_path, f"episode_{len(episodes) - 1:05d}.hdf5")

            init_hdf5(h5_path, episode_data["states"][0])

            steps = len(episode_data["states"])

            for i in range(steps):
                meta_data = episode_data["states"][i].copy()
                if i < len(episode_data["actions"]):
                    meta_data["action"] = episode_data["actions"][i]
                append_hdf5(h5_path, meta_data)
            print(
                "Success, states:",
                len(episode_data["states"]),
                "actions:",
                len(episode_data["actions"]),
            )
        else:
            print("Failed")

    sim_app.close()


if __name__ == "__main__":
    main()
