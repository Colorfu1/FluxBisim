from omni.isaac.kit import SimulationApp

sim_app = SimulationApp({'headless': True})

import argparse  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rootutils  # noqa: E402
import rospy  # noqa: E402
import yaml  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from sensor_msgs.msg import Image, JointState  # noqa: E402
from std_msgs.msg import Header, Int32, String  # noqa: E402

rootutils.setup_root(__file__, pythonpath=True)

from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

from envs import IsaacsimEnv  # noqa: E402

# (config_yaml_key, env_var_name, default_value) for each task type.
TASK_ENV_VAR_MAP = {
    'PickPlaceFruit': [
        ('fruit', 'FRUIT_NAME', 'banana'),
        ('plate', 'PLATE_NAME', 'plate'),
    ],
    'PassBook': [
        ('book', 'BOOK_NAME', 'book'),
        ('bookend', 'BOOKEND_NAME', 'bookend'),
    ],
    'RotatePitcherLid': [
        ('pitcher', 'PITCHER_NAME', 'pitcher'),
        ('pitcherlid', 'LID_NAME', 'pitcherlid'),
    ],
    'PullPushDrawer': [
        ('object', 'OBJECT_NAME', 'apple_drawer'),
        ('drawer', 'DRAWER_NAME', 'drawer'),
    ],
    'CloseBox': [
        ('nut', 'NUT_NAME', 'nut'),
        ('box', 'BOX_NAME', 'box'),
    ],
}

CAMERA_NAME_TO_TOPIC = {
    'left_cam': 'camera_l',
    'right_cam': 'camera_r',
    'head_cam': 'camera_f',
}


class IsaacSimROSBridge:
    """Bridge between Isaac Sim and ROS for evaluation."""

    def __init__(
        self,
        config_path: str,
        max_steps: int,
        num_episodes: int,
        save_images: bool,
        chunk_size: int,
    ):
        rospy.init_node('isaacsim_ros_bridge', anonymous=True)
        self.rate = rospy.Rate(10)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.config.setdefault('physics_dt', 1.0 / 120.0)
        self.config.setdefault('rendering_dt', 1.0 / 30.0)

        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.save_images = save_images
        self.chunk_size = chunk_size

        self.env = IsaacsimEnv(self.config)
        self.env.reset()

        for _ in range(60):
            self.env.world.step(render=True)

        self.cv_bridge = CvBridge()

        self.head_cam_frame_count = 0
        self.current_episode = -1
        self.head_cam_save_dir = None
        self.head_cam_base_dir = None
        if self.save_images:
            self.head_cam_base_dir = os.path.join(os.getcwd(), 'saved_images',
                                                  'head_cam')
            os.makedirs(self.head_cam_base_dir, exist_ok=True)
            rospy.loginfo(f"Head camera save dir: {self.head_cam_base_dir}")

        self._setup_publishers()
        self._setup_subscribers()

        self.pending_joint_commands = {}
        self.current_chunk_id = None
        self.executed_chunk_actions = 0

        self.left_joint_names = [f"llink{i}" for i in range(1, 8)]
        self.right_joint_names = [f"rlink{i}" for i in range(1, 8)]

        rospy.loginfo('Isaac Sim ROS bridge initialized')

    # ------------------------------------------------------------------
    # ROS setup
    # ------------------------------------------------------------------

    def _setup_publishers(self):
        self.image_publishers = {}
        camera_cfg = self.config.get('cameras', {})
        for cam_name in camera_cfg:
            topic_name = CAMERA_NAME_TO_TOPIC.get(cam_name, cam_name)
            rgb_topic = f"/{topic_name}/color/image_raw"
            self.image_publishers[cam_name] = rospy.Publisher(rgb_topic,
                                                              Image,
                                                              queue_size=1)
            rospy.loginfo(f"Image publisher: {rgb_topic} ({cam_name})")

        self.joint_left_publisher = rospy.Publisher('/puppet/joint_left',
                                                    JointState,
                                                    queue_size=1)
        self.joint_right_publisher = rospy.Publisher('/puppet/joint_right',
                                                     JointState,
                                                     queue_size=1)

        self.env_reset_publisher = rospy.Publisher('/env/reset',
                                                   String,
                                                   queue_size=1)
        self.chunk_done_publisher = rospy.Publisher('/env/chunk_done',
                                                    Int32,
                                                    queue_size=1)

    def _setup_subscribers(self):
        self.joint_left_subscriber = rospy.Subscriber(
            '/master/joint_left',
            JointState,
            self._joint_left_callback,
            queue_size=100,
            buff_size=524288,
        )
        self.joint_right_subscriber = rospy.Subscriber(
            '/master/joint_right',
            JointState,
            self._joint_right_callback,
            queue_size=100,
            buff_size=524288,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release all ROS and simulation resources."""
        try:
            rospy.loginfo('Cleaning up bridge resources ...')

            for cam_name, pub in self.image_publishers.items():
                try:
                    pub.unregister()
                except Exception as e:
                    rospy.logwarn(
                        f"Failed to close image publisher {cam_name}: {e}")

            for pub in (
                    self.joint_left_publisher,
                    self.joint_right_publisher,
                    self.chunk_done_publisher,
            ):
                try:
                    pub.unregister()
                except Exception as e:
                    rospy.logwarn(f"Failed to close joint publisher: {e}")

            if hasattr(self, 'env_reset_publisher'):
                try:
                    self.env_reset_publisher.unregister()
                except Exception as e:
                    rospy.logwarn(f"Failed to close reset publisher: {e}")

            for attr in (
                    'joint_left_subscriber',
                    'joint_right_subscriber',
            ):
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).unregister()
                    except Exception as e:
                        rospy.logwarn(
                            f"Failed to close subscriber {attr}: {e}")

            if hasattr(self, 'env') and self.env is not None:
                try:
                    if hasattr(self.env, 'cleanup'):
                        self.env.cleanup()
                except Exception as e:
                    rospy.logwarn(f"Failed to clean up Isaac Sim env: {e}")

            rospy.loginfo('Bridge cleanup complete')

        except Exception as e:
            rospy.logerr(f"Error during cleanup: {e}")
            rospy.logerr(traceback.format_exc())

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        return image

    def publish_images(self, obs, timestamp=None):
        """Publish RGB images from all cameras."""
        if timestamp is None:
            timestamp = rospy.Time.now()

        camera_cfg = self.config.get('cameras', {})
        for cam_name in camera_cfg:
            rgb_key = camera_cfg[cam_name].get('name', cam_name)
            if rgb_key not in obs or obs[rgb_key] is None:
                continue
            try:
                rgb_image = self._ensure_uint8(obs[rgb_key].copy())
                ros_image = self.cv_bridge.cv2_to_imgmsg(rgb_image, 'rgb8')
                ros_image.header.frame_id = f"{cam_name}_optical_frame"
                ros_image.header.stamp = timestamp
                self.image_publishers[cam_name].publish(ros_image)
            except Exception as e:
                rospy.logwarn(f"Failed to publish image {cam_name}: {e}")
                rospy.logwarn(traceback.format_exc())

    def save_head_cam_image(self, obs):
        """Save the head camera image to disk (if enabled)."""
        if not self.save_images:
            return

        if self.head_cam_save_dir is None:
            self.current_episode = 0
            self.head_cam_save_dir = os.path.join(
                self.head_cam_base_dir,
                f"episode_{self.current_episode:03d}")  # noqa: E231
            os.makedirs(self.head_cam_save_dir, exist_ok=True)
            self.head_cam_frame_count = 0

        try:
            camera_cfg = self.config.get('cameras', {})
            head_cam_cfg = camera_cfg.get('head_cam', {})
            rgb_key = head_cam_cfg.get('name', 'head_cam')
            if rgb_key in obs and obs[rgb_key] is not None:
                rgb_image = self._ensure_uint8(obs[rgb_key].copy())
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                filename = f"head_cam_{self.head_cam_frame_count:06d}.jpg"  # noqa: E231
                cv2.imwrite(os.path.join(self.head_cam_save_dir, filename),
                            bgr_image)
                self.head_cam_frame_count += 1
        except Exception as e:
            rospy.logwarn(f"Failed to save head_cam image: {e}")
            rospy.logwarn(traceback.format_exc())

    def _create_joint_state_msg(self, joint_names, dof_pos, timestamp,
                                frame_id):
        """Build a JointState message from dof_pos dict."""
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = timestamp
        joint_state.header.frame_id = frame_id
        joint_state.name = []
        joint_state.position = []
        joint_state.velocity = []
        joint_state.effort = []

        for joint_name in joint_names:
            if joint_name not in dof_pos:
                continue
            joint_state.name.append(joint_name)

            # link7 gripper conversion: publish (link7 - link8) * 0.1 / 0.07
            if joint_name.endswith('link7'):
                link8_name = joint_name.replace('link7', 'link8')
                if link8_name in dof_pos:
                    pos = ((float(dof_pos[joint_name]) -
                            float(dof_pos[link8_name])) * 0.1 / 0.07)
                else:
                    pos = float(dof_pos[joint_name])
            else:
                pos = float(dof_pos[joint_name])

            joint_state.position.append(pos)
            joint_state.velocity.append(0.0)
            joint_state.effort.append(0.0)

        return joint_state

    def publish_joint_states(self, obs, timestamp=None):
        """Publish left and right arm joint states."""
        try:
            if timestamp is None:
                timestamp = rospy.Time.now()

            robot_name = self.config.get('robot', 'piper')
            if robot_name not in obs or 'dof_pos' not in obs[robot_name]:
                return

            dof_pos = obs[robot_name]['dof_pos']
            left_msg = self._create_joint_state_msg(
                self.left_joint_names,
                dof_pos,
                timestamp,
                f"{robot_name}_left_base_link",
            )
            self.joint_left_publisher.publish(left_msg)

            right_msg = self._create_joint_state_msg(
                self.right_joint_names,
                dof_pos,
                timestamp,
                f"{robot_name}_right_base_link",
            )
            self.joint_right_publisher.publish(right_msg)

        except Exception as e:
            rospy.logwarn(f"Failed to publish joint states: {e}")
            rospy.logwarn(traceback.format_exc())

    def publish_chunk_done(self, chunk_id):
        """Notify the operator that the current action chunk is complete."""
        msg = Int32()
        msg.data = int(chunk_id)
        self.chunk_done_publisher.publish(msg)

    # ------------------------------------------------------------------
    # Joint state subscription
    # ------------------------------------------------------------------

    def _process_joint_callback(self, msg, arm_name, arm_key):
        """Decode incoming joint command and buffer it by action index."""
        try:
            if len(msg.position) < 7:
                rospy.logwarn(f"{arm_name} joint count insufficient")
                return

            if not msg.header.frame_id:
                rospy.logwarn(f"{arm_name} message missing frame_id")
                return

            # frame_id carries "chunk_id:action_idx" because ROS1 overwrites
            # header.seq with its own auto-increment counter.
            parts = msg.header.frame_id.split(':')
            if len(parts) != 2:
                rospy.logwarn(f"{arm_name} unexpected frame_id format: "
                              f"{msg.header.frame_id}")
                return

            chunk_id = int(parts[0])
            action_idx = int(parts[1])
            if action_idx < 0 or action_idx >= self.chunk_size:
                rospy.logwarn(
                    f"{arm_name} action index {action_idx} out of range for "
                    f"chunk size {self.chunk_size}")
                return

            if self.current_chunk_id is None:
                self.current_chunk_id = chunk_id
                self.executed_chunk_actions = 0
                self.pending_joint_commands.clear()
            elif chunk_id != self.current_chunk_id:
                rospy.logwarn(
                    f"Ignoring {arm_name} command for chunk {chunk_id}; "  # noqa: E702
                    f"current active chunk is {self.current_chunk_id}")
                return

            joint_positions = np.zeros(8)
            joint_positions[:6] = msg.position[:6]

            # Inverse gripper conversion: link7 = published * 0.07 / 0.2
            joint_positions[6] = msg.position[6] * 0.07 / 0.2
            joint_positions[7] = -joint_positions[6]

            action_commands = self.pending_joint_commands.setdefault(
                action_idx, {})
            action_commands[arm_key] = joint_positions.copy()
        except Exception as e:
            rospy.logwarn(f"Error processing {arm_name} joint state: {e}")
            rospy.logwarn(traceback.format_exc())

    def _joint_left_callback(self, msg):
        self._process_joint_callback(msg, 'left arm', 'left')

    def _joint_right_callback(self, msg):
        self._process_joint_callback(msg, 'right arm', 'right')

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_received_joint_states(self):
        """Apply the next joint command from the active chunk.

        Returns:
            (action_applied, chunk_completed, completed_chunk_id)
        """
        try:
            if self.current_chunk_id is None:
                return False, False, None

            action_idx = self.executed_chunk_actions
            action_commands = self.pending_joint_commands.get(action_idx)
            if action_commands is None:
                return False, False, None

            left_positions = action_commands.get('left')
            right_positions = action_commands.get('right')
            if left_positions is None or right_positions is None:
                return False, False, None

            action = [
                ArticulationAction(joint_positions=left_positions),
                ArticulationAction(joint_positions=right_positions),
            ]
            try:
                self.env.robot.apply_action(action)
                self.env.world.step(render=True)
            except Exception as e:
                rospy.logwarn(f"Failed to apply joint action: {e}")
                rospy.logwarn(traceback.format_exc())
                return False, False, None

            del self.pending_joint_commands[action_idx]
            self.executed_chunk_actions += 1

            chunk_completed = self.executed_chunk_actions >= self.chunk_size
            completed_chunk_id = self.current_chunk_id if chunk_completed else None
            if chunk_completed:
                self.pending_joint_commands.clear()
                self.current_chunk_id = None
                self.executed_chunk_actions = 0

            return True, chunk_completed, completed_chunk_id

        except Exception as e:
            rospy.logwarn(f"Error in apply_received_joint_states: {e}")
            rospy.logwarn(traceback.format_exc())
            return False, False, None

    # ------------------------------------------------------------------
    # Environment reset & task check
    # ------------------------------------------------------------------

    def _flush_joint_queues(self):
        self.pending_joint_commands.clear()
        self.current_chunk_id = None
        self.executed_chunk_actions = 0

    def reset_environment(self):
        """Reset the simulation environment and wait for it to stabilize."""
        try:
            rospy.loginfo('Resetting environment ...')
            self.env.reset()

            if self.save_images:
                self.current_episode += 1
                self.head_cam_save_dir = os.path.join(
                    self.head_cam_base_dir,
                    f"episode_{self.current_episode:03d}")  # noqa: E231
                os.makedirs(self.head_cam_save_dir, exist_ok=True)
                self.head_cam_frame_count = 0

            for _ in range(60):
                self.env.world.step(render=True)

            rospy.loginfo('Environment reset complete')

            self.env.get_observation()

            reset_msg = String()
            reset_msg.data = 'environment_reset'
            self.env_reset_publisher.publish(reset_msg)

            self._flush_joint_queues()

        except Exception as e:
            rospy.logerr(f"Error resetting environment: {e}")
            rospy.logerr(traceback.format_exc())

    def check_task_completion(self):
        """Return True if the current task checker reports success."""
        try:
            if hasattr(self.env, 'task') and hasattr(self.env.task, 'checker'):
                return self.env.task.checker.check(self.env)
            return False
        except Exception as e:
            rospy.logwarn(f"Error checking task completion: {e}")
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def print_statistics(self, success_list, task_completion_steps):
        """Print evaluation statistics."""
        if not success_list:
            rospy.logwarn('No statistics to display')
            return

        total = len(success_list)
        successes = sum(success_list)
        rate = successes / total if total > 0 else 0.0

        rospy.loginfo('=' * 60)
        rospy.loginfo('Evaluation Statistics:')
        rospy.loginfo(f"  Total episodes : {total}")
        rospy.loginfo(f"  Successes      : {successes}")
        rospy.loginfo(f"  Failures       : {total - successes}")
        rospy.loginfo(
            f"  Success rate   : {rate:.4f} ({rate * 100:.2f}%)"  # noqa: E231
        )

        if task_completion_steps:
            mean_steps = np.mean(task_completion_steps)
            rospy.loginfo(f"  Mean steps     : {mean_steps:.2f}")  # noqa: E231
            if len(task_completion_steps) > 1:
                std_steps = np.std(task_completion_steps)
                rospy.loginfo(
                    f"  Std steps      : {std_steps:.2f}")  # noqa: E231

        rospy.loginfo('=' * 60)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Run the evaluation loop."""
        rospy.loginfo(f"Starting evaluation: max_steps={self.max_steps}, "
                      f"num_episodes={self.num_episodes}")

        episode = 0
        step = 0

        success_list = []
        task_completion_steps = []
        statistics_printed = False

        while not rospy.is_shutdown() and sim_app.is_running():
            try:
                action_applied, chunk_completed, completed_chunk_id = (
                    self.apply_received_joint_states())
                obs = self.env.get_observation()
                timestamp = None

                if action_applied:
                    self.save_head_cam_image(obs)
                    step += 1

                if chunk_completed:
                    if self.check_task_completion():
                        rospy.loginfo(
                            f"Episode {episode} succeeded, steps: {step}")
                        task_completion_steps.append(step)
                        success_list.append(True)

                        self.reset_environment()
                        episode += 1
                        step = 0

                        if episode >= self.num_episodes:
                            self.print_statistics(success_list,
                                                  task_completion_steps)
                            statistics_printed = True
                            break

                        self.rate.sleep()
                        continue

                    if step >= self.max_steps:
                        msg = (
                            f"Episode {episode} failed: reached max steps {self.max_steps}"
                        )
                        rospy.logwarn(msg)
                        print(msg, flush=True)
                        success_list.append(False)

                        self.reset_environment()
                        episode += 1
                        step = 0

                        if episode >= self.num_episodes:
                            self.print_statistics(success_list,
                                                  task_completion_steps)
                            statistics_printed = True
                            break

                        self.rate.sleep()
                        continue

                    timestamp = rospy.Time.now()
                    self.publish_images(obs, timestamp)
                    self.publish_joint_states(obs, timestamp)
                    self.publish_chunk_done(completed_chunk_id)

                if timestamp is None:
                    timestamp = rospy.Time.now()
                    self.publish_images(obs, timestamp)
                    self.publish_joint_states(obs, timestamp)

                self.rate.sleep()

            except rospy.ROSInterruptException:
                rospy.loginfo('ROS interrupt received, exiting ...')
                break
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                rospy.logerr(traceback.format_exc())
                self.rate.sleep()

        if success_list and not statistics_printed:
            rospy.loginfo('Printing statistics before exit:')
            self.print_statistics(success_list, task_completion_steps)

        rospy.loginfo('Main loop finished')


# ----------------------------------------------------------------------
# Top-level helpers
# ----------------------------------------------------------------------


def signal_handler(sig, frame):
    rospy.loginfo(f"Received signal {sig}, shutting down ...")
    if not rospy.is_shutdown():
        rospy.signal_shutdown('Signal received')


def cleanup_resources(bridge=None):
    """Clean up all resources (bridge, ROS, Isaac Sim)."""
    rospy.loginfo('Cleaning up resources ...')

    if bridge is not None:
        try:
            bridge.cleanup()
        except Exception as e:
            rospy.logwarn(f"Unexpected error during bridge cleanup: {e}")

    try:
        if not rospy.is_shutdown():
            rospy.signal_shutdown('Program exit')
    except Exception as e:
        rospy.logwarn(f"Error shutting down ROS: {e}")

    try:
        if sim_app.is_running():
            sim_app.close()
    except Exception as e:
        rospy.logwarn(f"Error closing Isaac Sim: {e}")

    rospy.loginfo('Resource cleanup complete')


def setup_task_env_vars(config):
    """Set task-specific environment variables based on the config."""
    task_name = config.get('task')
    if task_name not in TASK_ENV_VAR_MAP:
        rospy.logwarn(f"Unknown task '{task_name}', skipping task env vars. "
                      f"Known tasks: {list(TASK_ENV_VAR_MAP.keys())}")
        return

    for cfg_key, env_var, default in TASK_ENV_VAR_MAP[task_name]:
        value = config.get(cfg_key, default)
        os.environ[env_var] = value
        rospy.loginfo(f"Set {env_var}={value}")


def main():
    parser = argparse.ArgumentParser(
        description='Isaac Sim & ROS evaluation script')
    parser.add_argument(
        '--env',
        type=str,
        required=True,
        help='Environment name, mapped to TABLE_NAME=<env>_table and '
        'BASE_NAME=base_<env>. Choices: kitchen, apartment, '
        'apartmentshort, industry',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Config file path (relative to project root or absolute)',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1600,
        help='Maximum steps per episode',
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=50,
        help='Number of episodes to run',
    )
    parser.add_argument(
        '--save-images',
        type=lambda x: str(x).lower() == 'true',
        default=False,
        help='Whether to save head_camera images',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=32,
        help='Fixed number of actions in each VLA action chunk',
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    if not os.path.exists(config_path):
        rospy.logerr(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        rospy.logerr(f"Failed to read config file: {e}")
        sys.exit(1)

    os.environ['TABLE_NAME'] = f"{args.env}_table"
    os.environ['BASE_NAME'] = f"base_{args.env}"
    rospy.loginfo(
        f"Environment: {args.env} -> TABLE_NAME={os.environ['TABLE_NAME']}, "
        f"BASE_NAME={os.environ['BASE_NAME']}")

    setup_task_env_vars(config)

    bridge = None
    try:
        rospy.loginfo('Initializing Isaac Sim ROS bridge ...')
        bridge = IsaacSimROSBridge(
            config_path,
            max_steps=args.max_steps,
            num_episodes=args.num_episodes,
            save_images=args.save_images,
            chunk_size=args.chunk_size,
        )
        bridge.run()
    except KeyboardInterrupt:
        rospy.loginfo('Interrupted by user, exiting ...')
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        rospy.logerr(traceback.format_exc())
    finally:
        cleanup_resources(bridge)


if __name__ == '__main__':
    main()
