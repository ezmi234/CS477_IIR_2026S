"""
Motion node placeholder.

Subscribes to /motion_request and publishes a dummy /motion_result.
Supports configurable failure simulation for Phase 3.
"""

import random

import rclpy
from rclpy.node import Node
from team01_solution_msgs.msg import MotionResult, Task


class MotionNode(Node):

    def __init__(self):
        super().__init__('motion_node')

        self.declare_parameter('failure_rate', 0.2)
        self.declare_parameter('always_fail', False)

        self.failure_rate = self.get_parameter('failure_rate').value
        self.always_fail = self.get_parameter('always_fail').value

        self.publisher = self.create_publisher(MotionResult, '/motion_result', 10)
        self.create_subscription(Task, '/motion_request', self.request_callback, 10)

        self.get_logger().info('Motion node ready and listening on /motion_request')
        self.get_logger().info(f'failure_rate={self.failure_rate} always_fail={self.always_fail}')

    def request_callback(self, msg: Task):
        self.get_logger().info(f'Received /motion_request item_id={msg.item_id} target={msg.target_location}')

        failed = self.always_fail or random.random() < float(self.failure_rate)
        result = MotionResult()

        if failed:
            result.success = False
            result.status = 'motion_failed'
            self.get_logger().warn(f'Motion failed for item_id={msg.item_id}')
        else:
            result.success = True
            result.status = 'motion_done'
            self.get_logger().info(f'Motion succeeded for item_id={msg.item_id}')

        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
