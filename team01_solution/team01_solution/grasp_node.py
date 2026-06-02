"""
Grasp node placeholder.

Subscribes to /grasp_request and publishes a dummy /grasp_result.
Supports configurable failure simulation for Phase 3.
"""

import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from team01_solution_msgs.msg import GraspResult, Task


class GraspNode(Node):

    def __init__(self):
        super().__init__('grasp_node')

        self.declare_parameter('failure_rate', 0.2)
        self.declare_parameter('always_fail', False)

        self.failure_rate = self.get_parameter('failure_rate').value
        self.always_fail = self.get_parameter('always_fail').value

        self.publisher = self.create_publisher(GraspResult, '/grasp_result', 10)
        self.create_subscription(Task, '/grasp_request', self.request_callback, 10)

        self.get_logger().info('Grasp node ready and listening on /grasp_request')
        self.get_logger().info(f'failure_rate={self.failure_rate} always_fail={self.always_fail}')

    def make_grasp_pose(self):
        pose = Pose()
        pose.position = Point(x=0.05, y=0.0, z=0.10)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose

    def request_callback(self, msg: Task):
        self.get_logger().info(f'Received /grasp_request item_id={msg.item_id} target={msg.target_location}')

        failed = self.always_fail or random.random() < float(self.failure_rate)
        result = GraspResult()
        result.item_id = msg.item_id

        if failed:
            result.success = False
            result.grasp_pose = Pose()
            self.get_logger().warn(f'Grasp failed for item_id={msg.item_id}')
        else:
            result.success = True
            result.grasp_pose = self.make_grasp_pose()
            self.get_logger().info(f'Grasp succeeded for item_id={msg.item_id}')

        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
