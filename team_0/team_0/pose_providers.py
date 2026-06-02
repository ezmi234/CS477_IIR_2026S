import rclpy
from geometry_msgs.msg import Pose
from riro_srvs.srv import StringPose

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
