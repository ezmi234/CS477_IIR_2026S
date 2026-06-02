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
        while rclpy.ok() and not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for /get_object_pose service...')
            rclpy.spin_once(self.node, timeout_sec=0.1)

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
        while rclpy.ok() and not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for detect_objects_with_prompt service...')
            rclpy.spin_once(self.node, timeout_sec=0.1)

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

    The vision server exposes the same StringPose service as Team 01 detection,
    but the response Pose has no frame_id. It also publishes
    /vision/selected_pose as PoseStamped. This provider uses the stamped pose
    whenever available and transforms it to the executor base frame.
    """

    def __init__(self, node, client):
        super().__init__(node)
        self.client = client
        self.latest_pose_stamped = None
        self.latest_detection_json = ''
        pose_topic = self.node.get_parameter('vision_pose_topic').value
        detection_topic = self.node.get_parameter('vision_detection_topic').value
        self.node.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)
        self.node.create_subscription(String, detection_topic, self.detection_callback, 10)

    def pose_callback(self, msg):
        self.latest_pose_stamped = msg

    def detection_callback(self, msg):
        self.latest_detection_json = msg.data

    def wait_until_ready(self):
        while rclpy.ok() and not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for external vision service...')
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def get_object_pose(self, object_name):
        prompt_name = object_name.replace('_', ' ')
        request = StringPose.Request()
        request.data = f'Detect a {prompt_name} and return pose'
        self.latest_pose_stamped = None
        self.latest_detection_json = ''
        self.node.get_logger().info(f'Vision prompt: {request.data}')

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=20.0)
        if not future.done() or future.result() is None:
            return None

        response_pose = future.result().pose
        if (
            abs(response_pose.position.x) < 1e-9
            and abs(response_pose.position.y) < 1e-9
            and abs(response_pose.position.z) < 1e-9
            and abs(response_pose.orientation.w) < 1e-9
        ):
            self.node.get_logger().error(f'Vision service returned no pose for {object_name}.')
            return None

        timeout_sec = float(self.node.get_parameter('vision_pose_timeout').value)
        start_time = self.node.get_clock().now().nanoseconds * 1e-9
        while rclpy.ok() and self.latest_pose_stamped is None:
            now = self.node.get_clock().now().nanoseconds * 1e-9
            if now - start_time > timeout_sec:
                break
            rclpy.spin_once(self.node, timeout_sec=0.05)

        if self.latest_pose_stamped is not None and self.latest_pose_stamped.header.frame_id:
            pose_stamped = self.latest_pose_stamped
            self.node.get_logger().info(
                'Vision selected pose: '
                f'frame={pose_stamped.header.frame_id}, '
                f'x={pose_stamped.pose.position.x:.3f}, '
                f'y={pose_stamped.pose.position.y:.3f}, '
                f'z={pose_stamped.pose.position.z:.3f}'
            )
            if self.latest_detection_json:
                self.node.get_logger().info(
                    f'Vision selected detection: {self.latest_detection_json[:500]}'
                )
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
