"""
Motion node placeholder.

Subscribes to /motion_request and publishes a dummy /motion_result.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MotionNode(Node):

    def __init__(self):
        super().__init__('motion_node')
        self.publisher = self.create_publisher(String, '/motion_result', 10)
        self.create_subscription(String, '/motion_request', self.request_callback, 10)
        self.get_logger().info('Motion node ready and listening on /motion_request')

    def request_callback(self, msg: String):
        request = msg.data.strip()
        self.get_logger().info(f'Received motion request: {request}')

        result = 'motion_done'
        self.publisher.publish(String(data=result))
        self.get_logger().info(f'Published motion result: {result}')


def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
