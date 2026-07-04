# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            rename_map=config.rename_map,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Exponential backoff state for server unavailability
        self._backoff_until = 0.0  # timestamp until which we should back off
        self._backoff_duration = 0.0  # current backoff duration in seconds

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # Recording setup (only when record_dataset_repo_id is provided)
        self.dataset = None
        self._video_encoding_manager = None
        self._robot_observation_processor = None
        self._last_obs_processed = None
        self._listener = None
        self._events = None

        if config.record_dataset_repo_id is not None:
            self._setup_recording()

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _setup_recording(self):
        """Set up dataset recording using the same flow as lerobot-record."""
        from lerobot.common.control_utils import init_keyboard_listener
        from lerobot.datasets import LeRobotDataset
        from lerobot.datasets.pipeline_features import (
            aggregate_pipeline_dataset_features,
            create_initial_features,
        )
        from lerobot.datasets.video_utils import VideoEncodingManager
        from lerobot.processor.factory import make_default_processors
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

        # Reuse official processor and feature construction (lerobot_record.py:451-466)
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )
        self._robot_observation_processor = robot_observation_processor
        self._build_dataset_frame = build_dataset_frame
        self._OBS_STR = OBS_STR
        self._ACTION = ACTION

        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=self.robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=self.robot.observation_features),
                use_videos=True,
            ),
        )

        # Reuse official dataset creation (lerobot_record.py:491-505)
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.record_dataset_repo_id,
            fps=self.config.fps,
            root=self.config.record_dataset_root,
            robot_type=self.robot.name,
            features=dataset_features,
            use_videos=True,
            streaming_encoding=True,
        )

        self._video_encoding_manager = VideoEncodingManager(self.dataset)
        self._video_encoding_manager.__enter__()

        # Reuse official keyboard listener (lerobot_record.py:526)
        self._listener, self._events = init_keyboard_listener()

        self.logger.info(
            f"Recording enabled: repo_id={self.config.record_dataset_repo_id}, "
            f"num_episodes={self.config.record_num_episodes}"
        )
        self.logger.info("Keyboard controls: RIGHT=save episode, LEFT=rerecord episode, ESC=stop recording")

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        # Finalize recording dataset before disconnecting
        if self.dataset is not None:
            try:
                if self._video_encoding_manager is not None:
                    self._video_encoding_manager.__exit__(None, None, None)
                self.dataset.finalize()
                self.logger.info(f"Dataset finalized at: {self.dataset.root}")
            except Exception as e:
                self.logger.warning(f"Dataset finalize error (non-fatal): {e}")

        # Stop keyboard listener
        if self._listener is not None:
            self._listener.stop()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        if self._is_backing_off():
            return False

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            self._record_success()
            return True

        except grpc.RpcError:
            self._record_failure()
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            if self._is_backing_off():
                time.sleep(1.0)
                continue

            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError:
                self._record_failure()

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _is_backing_off(self) -> bool:
        """Check if we're currently in a backoff period."""
        return time.time() < self._backoff_until

    def _record_failure(self) -> None:
        """Record a connection failure and increase backoff duration."""
        if self._backoff_duration == 0.0:
            self._backoff_duration = 1.0
        else:
            self._backoff_duration = min(self._backoff_duration * 2, 30.0)
        self._backoff_until = time.time() + self._backoff_duration
        self.logger.warning(f"Server unavailable, backing off for {self._backoff_duration:.1f}s")

    def _record_success(self) -> None:
        """Reset backoff state on successful communication."""
        if self._backoff_duration > 0.0:
            self.logger.info("Server connection restored")
        self._backoff_duration = 0.0
        self._backoff_until = 0.0

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        try:
            _performed_action = self.robot.send_action(
                self._action_tensor_to_action_dict(timed_action.get_action())
            )
        except ConnectionError as e:
            self.logger.error(f"Robot connection lost: {e}. Stopping control loop.")
            self.shutdown_event.set()
            raise

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        # Record frame (observation + action) to dataset
        if self.dataset is not None and self._last_obs_processed is not None:
            try:
                self._record_frame(_performed_action)
            except Exception as e:
                self.logger.warning(f"Recording error (non-fatal): {e}")

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(
        self, task: str, verbose: bool = False, raw_observation: RawObservation | None = None
    ) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            if raw_observation is None:
                raw_observation = self.robot.get_observation()
                raw_observation["task"] = task

            # Store observation for recording (reuse official processor)
            if self.dataset is None:
                self._last_obs_processed = self._robot_observation_processor(raw_observation)

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def _record_frame(self, performed_action: dict[str, Any]):
        """Record a single frame (observation + action) to the dataset.

        Reuses the official build_dataset_frame from lerobot-record.
        """
        observation_frame = self._build_dataset_frame(
            self.dataset.features, self._last_obs_processed, prefix=self._OBS_STR
        )
        action_frame = self._build_dataset_frame(self.dataset.features, performed_action, prefix=self._ACTION)
        frame = {**observation_frame, **action_frame, "task": self.config.task}
        self.dataset.add_frame(frame)

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None
        recorded_episodes = 0

        while self.running:
            control_loop_start = time.perf_counter()

            # Slow down loop when server is unavailable
            if self._is_backing_off():
                time.sleep(1.0)
                continue

            """Control loop: (0) Capture camera frame once per iteration for recording & server"""
            _send_to_server = self._ready_to_send_observation()
            _raw_obs = None
            if self.dataset is not None or _send_to_server:
                try:
                    _raw_obs = self.robot.get_observation()
                    _raw_obs["task"] = task
                    # Store for recording
                    if self.dataset is not None:
                        self._last_obs_processed = self._robot_observation_processor(_raw_obs)
                except Exception as e:
                    self.logger.warning(f"Observation capture error (non-fatal): {e}")
                    _raw_obs = None

            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if _send_to_server and _raw_obs is not None:
                _captured_observation = self.control_loop_observation(task, verbose, raw_observation=_raw_obs)

            """Control loop: (3) Recording episode management via keyboard"""
            if self.dataset is not None and self._events is not None and self._events["exit_early"]:
                self._events["exit_early"] = False
                if self._events.get("rerecord_episode", False):
                    # Left arrow: discard current episode and re-record
                    self._events["rerecord_episode"] = False
                    try:
                        self.dataset.clear_episode_buffer()
                        self.logger.info("Episode discarded, re-recording...")
                    except Exception as e:
                        self.logger.warning(f"Clear episode error (non-fatal): {e}")
                else:
                    # Right arrow: save current episode
                    try:
                        self.dataset.save_episode()
                        recorded_episodes += 1
                        self.logger.info(
                            f"Episode saved. Total: {recorded_episodes}/{self.config.record_num_episodes}"
                        )
                    except Exception as e:
                        self.logger.warning(f"Episode save error (non-fatal): {e}")

                if self._events.get("stop_recording", False):
                    # Escape: stop recording but keep control loop running
                    self._events["stop_recording"] = False
                    try:
                        if self._video_encoding_manager is not None:
                            self._video_encoding_manager.__exit__(None, None, None)
                        self.dataset.finalize()
                        self.logger.info(f"Recording stopped. Dataset finalized at: {self.dataset.root}")
                    except Exception as e:
                        self.logger.warning(f"Dataset finalize error (non-fatal): {e}")
                    self.dataset = None

            # Check if target episodes reached
            if self.dataset is not None and recorded_episodes >= self.config.record_num_episodes:
                self.logger.info(f"Target episodes reached ({recorded_episodes}). Stopping recording.")
                try:
                    if self._video_encoding_manager is not None:
                        self._video_encoding_manager.__exit__(None, None, None)
                    self.dataset.finalize()
                    self.logger.info(f"Dataset finalized at: {self.dataset.root}")
                except Exception as e:
                    self.logger.warning(f"Dataset finalize error (non-fatal): {e}")
                self.dataset = None

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
