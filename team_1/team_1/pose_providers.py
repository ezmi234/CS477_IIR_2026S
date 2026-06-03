import json
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from riro_srvs.srv import StringPose
from std_msgs.msg import String

from .config import HARDCODED_PICK_TARGETS


class PoseProvider:
    def __init__(self, node):
        self.node = node

    def wait_until_ready(self):
        return

    def get_object_pose(self, object_name):
        raise NotImplementedError


class HardcodedPoseProvider(PoseProvider):
    def get_object_pose(self, object_name):
        target = HARDCODED_PICK_TARGETS.get(object_name)
        if target is None:
            self.node.get_logger().error(f'No hardcoded pick target for {object_name}.')
            return None

        pose = Pose()
        pose.position.x = float(target[0])
        pose.position.y = float(target[1])
        pose.position.z = float(target[2])
        pose.orientation.w = 1.0
        self.node.get_logger().warn(
            'Using hardcoded pick target. Restart/reset the simulator before repeat trials.'
        )
        self.node.get_logger().info(
            'Hardcoded target: '
            f'x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}'
        )
        return pose


class GroundTruthPoseProvider(PoseProvider):
    def __init__(self, node, client):
        super().__init__(node)
        self.client = client

    def wait_until_ready(self):
        deadline = time.monotonic() + float(self.node.get_parameter('startup_wait_timeout').value)
        while (
            rclpy.ok()
            and not self.client.wait_for_service(timeout_sec=1.0)
            and time.monotonic() < deadline
        ):
            self.node.get_logger().info('Waiting for /get_object_pose service...')
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if not self.client.service_is_ready():
            self.node.get_logger().warn(
                '/get_object_pose was not ready before startup timeout. '
                'Ground-truth debug requests may fail.'
            )

    def get_object_pose(self, object_name):
        self.node.get_logger().warn(
            'Using /get_object_pose debug fallback. Do not use this mode in competition.'
        )
        request = StringPose.Request()
        request.data = object_name
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return None

        pose = future.result().pose
        pose.position.z += self.node.ground_truth_z_offset(object_name)
        pose.orientation.w = 1.0
        self.node.get_logger().info(
            'Ground-truth target: '
            f'x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}'
        )
        return pose


class DetectionPoseProvider(PoseProvider):
    def __init__(self, node, client):
        super().__init__(node)
        self.client = client

    def wait_until_ready(self):
        deadline = time.monotonic() + float(self.node.get_parameter('startup_wait_timeout').value)
        while (
            rclpy.ok()
            and not self.client.wait_for_service(timeout_sec=1.0)
            and time.monotonic() < deadline
        ):
            self.node.get_logger().info('Waiting for detect_objects_with_prompt service...')
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if not self.client.service_is_ready():
            self.node.get_logger().warn(
                'detect_objects_with_prompt was not ready before startup timeout. '
                'Detection requests may fail.'
            )

    def get_object_pose(self, object_name):
        prompt_name = object_name.replace('_', ' ')
        request = StringPose.Request()
        request.data = f'Detect a {prompt_name} and return [ymin, xmin, ymax, xmax, label]'
        self.node.get_logger().info(f'Detection prompt: {request.data}')

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=20.0)
        if not future.done() or future.result() is None:
            return None

        pose_camera = future.result().pose
        if (
            abs(pose_camera.position.x) < 1e-9
            and abs(pose_camera.position.y) < 1e-9
            and abs(pose_camera.position.z) < 1e-9
            and abs(pose_camera.orientation.w) < 1e-9
        ):
            self.node.get_logger().error(f'Detection service returned no pose for {object_name}.')
            return None
        return self.node.transform_pose(
            pose_camera,
            self.node.get_parameter('camera_frame').value,
            self.node.get_parameter('base_frame').value,
        )


class VisionPoseProvider(PoseProvider):
    """Adapter for ezmi234 feat/vision output.

    The vision server exposes the same StringPose service as the detector,
    but the response Pose has no frame_id. It also publishes the selected object
    center and an RGB-D grasp estimate. This provider prefers the base-frame
    grasp and falls back to the stamped object center when needed.
    """

    def __init__(self, node, client):
        super().__init__(node)
        self.client = client
        self.latest_pose_stamped = None
        self.latest_grasp_base = None
        self.latest_detection_json = ''
        self.latest_grasp_json = ''
        pose_topic = self.node.get_parameter('vision_pose_topic').value
        detection_topic = self.node.get_parameter('vision_detection_topic').value
        grasp_base_topic = self.node.get_parameter('vision_grasp_base_topic').value
        grasp_candidates_topic = self.node.get_parameter('vision_grasp_candidates_topic').value
        self.node.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)
        self.node.create_subscription(String, detection_topic, self.detection_callback, 10)
        self.node.create_subscription(PoseStamped, grasp_base_topic, self.grasp_base_callback, 10)
        self.node.create_subscription(String, grasp_candidates_topic, self.grasp_candidates_callback, 10)

    def pose_callback(self, msg):
        self.latest_pose_stamped = msg

    def grasp_base_callback(self, msg):
        self.latest_grasp_base = msg

    def detection_callback(self, msg):
        self.latest_detection_json = msg.data

    def grasp_candidates_callback(self, msg):
        self.latest_grasp_json = msg.data

    def wait_until_ready(self):
        deadline = time.monotonic() + float(self.node.get_parameter('startup_wait_timeout').value)
        while (
            rclpy.ok()
            and not self.client.wait_for_service(timeout_sec=1.0)
            and time.monotonic() < deadline
        ):
            self.node.get_logger().info('Waiting for external vision service...')
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if not self.client.service_is_ready():
            self.node.get_logger().warn(
                'Vision service was not ready before startup timeout. '
                'The executor will stay in standby, but command execution may fail.'
            )

    def get_object_pose(self, object_name):
        prompt_name = object_name.replace('_', ' ')
        response_pose = None
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            request = StringPose.Request()
            request.data = f'Detect a {prompt_name} and return pose'
            self.latest_pose_stamped = None
            self.latest_grasp_base = None
            self.latest_detection_json = ''
            self.latest_grasp_json = ''
            self.node.get_logger().info(
                f'Vision prompt ({attempt}/{max_attempts}): {request.data}'
            )

            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=20.0)
            if not future.done() or future.result() is None:
                self.node.get_logger().warn(
                    f'Vision service did not return for {object_name} on attempt {attempt}.'
                )
            else:
                response_pose = future.result().pose
                if not self._is_zero_pose(response_pose):
                    break
                self.node.get_logger().warn(
                    f'Vision service returned no pose for {object_name} on attempt {attempt}.'
                )

            if attempt < max_attempts:
                time.sleep(0.3)

        if response_pose is None or self._is_zero_pose(response_pose):
            self.node.get_logger().error(
                f'Vision failed to detect {object_name} after {max_attempts} attempts.'
            )
            return None

        timeout_sec = float(self.node.get_parameter('vision_pose_timeout').value)
        start_time = time.monotonic()
        while rclpy.ok() and self.latest_grasp_base is None:
            now = time.monotonic()
            if now - start_time > timeout_sec:
                break
            rclpy.spin_once(self.node, timeout_sec=0.05)

        if self.latest_pose_stamped is not None and self.latest_pose_stamped.header.frame_id:
            pose_stamped = self.latest_pose_stamped
            self.node.get_logger().info(
                'Vision selected object pose: '
                f'frame={pose_stamped.header.frame_id}, '
                f'x={pose_stamped.pose.position.x:.3f}, '
                f'y={pose_stamped.pose.position.y:.3f}, '
                f'z={pose_stamped.pose.position.z:.3f}'
            )

        if self.latest_detection_json:
            self.log_selected_detection()

        if self.latest_grasp_base is not None and self.latest_grasp_base.header.frame_id:
            pose_stamped = self.latest_grasp_base
            score = None
            try:
                score = json.loads(self.latest_grasp_json).get('grasp_score')
            except Exception:
                score = None
            score_text = f', score={float(score):.3f}' if score is not None else ''
            self.node.get_logger().info(
                'Selected grasp: '
                f'frame={pose_stamped.header.frame_id}, '
                f'x={pose_stamped.pose.position.x:.3f}, '
                f'y={pose_stamped.pose.position.y:.3f}, '
                f'z={pose_stamped.pose.position.z:.3f}'
                f'{score_text}'
            )
            if pose_stamped.header.frame_id == self.node.get_parameter('base_frame').value:
                return pose_stamped.pose
            transformed = self.node.transform_pose_stamped(
                pose_stamped,
                self.node.get_parameter('base_frame').value,
            )
            if transformed is not None:
                return transformed
            self.node.get_logger().warn(
                'Vision grasp pose was available but could not be transformed; '
                'falling back to object center pose.'
            )

        if self.latest_pose_stamped is not None and self.latest_pose_stamped.header.frame_id:
            pose_stamped = self.latest_pose_stamped
            self.node.get_logger().warn('No grasp pose available; falling back to object center pose.')
            return self.node.transform_pose_stamped(
                pose_stamped,
                self.node.get_parameter('base_frame').value,
            )

        fallback_frame = self.node.get_parameter('vision_fallback_frame').value
        self.node.get_logger().warn(
            'Vision selected PoseStamped was not received; '
            f'falling back to response.pose in frame {fallback_frame!r}.'
        )
        return self.node.transform_pose(
            response_pose,
            fallback_frame,
            self.node.get_parameter('base_frame').value,
        )

    @staticmethod
    def _is_zero_pose(pose):
        return (
            abs(pose.position.x) < 1e-9
            and abs(pose.position.y) < 1e-9
            and abs(pose.position.z) < 1e-9
        )

    def log_selected_detection(self):
        try:
            data = json.loads(self.latest_detection_json)
            self.node.get_logger().info(
                'Selected detection: '
                f"label={data.get('label', '')}, "
                f"camera={data.get('camera_name', '')}, "
                f"score={float(data.get('score', 0.0)):.3f}"
            )
        except Exception:
            self.node.get_logger().info(
                f'Vision selected detection: {self.latest_detection_json[:500]}'
            )
