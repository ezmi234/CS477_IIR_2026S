"""
Detection node placeholder.

Subscribes to /detect_request and publishes a dummy /detect_result.
Supports configurable failure simulation for Phase 3.
"""

import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from team01_solution_msgs.msg import DetectionResult, Task


class DetectionNode(Node):

    def __init__(self):
        super().__init__('detection_node')

        self.declare_parameter('failure_rate', 0.2)
        self.declare_parameter('always_fail', False)

        self.failure_rate = self.get_parameter('failure_rate').value
        self.always_fail = self.get_parameter('always_fail').value

        self.publisher = self.create_publisher(DetectionResult, '/detect_result', 10)
        self.create_subscription(Task, '/detect_request', self.request_callback, 10)

        self.get_logger().info('Detection node ready and listening on /detect_request')
        self.get_logger().info(f'failure_rate={self.failure_rate} always_fail={self.always_fail}')

    def make_pose(self):
        pose = Pose()
        pose.position = Point(x=0.15, y=0.0, z=0.05)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose

    def request_callback(self, msg: Task):
        self.get_logger().info(f'Received /detect_request item_id={msg.item_id} target={msg.target_location}')

        failed = self.always_fail or random.random() < float(self.failure_rate)
        result = DetectionResult()
        result.item_id = msg.item_id

        if failed:
            result.success = False
            result.confidence = 0.0
            result.pose = Pose()
            self.get_logger().warn(f'Detection failed for item_id={msg.item_id}')
        else:
            result.success = True
            result.confidence = float(random.uniform(0.7, 1.0))
            result.pose = self.make_pose()
            self.get_logger().info(f'Detection succeeded for item_id={msg.item_id} confidence={result.confidence:.2f}')

        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
