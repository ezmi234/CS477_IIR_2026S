"""
Grasp node placeholder.

Subscribes to /grasp_request and publishes a dummy /grasp_result.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GraspNode(Node):

    def __init__(self):
        super().__init__('grasp_node')
        self.publisher = self.create_publisher(String, '/grasp_result', 10)
        self.create_subscription(String, '/grasp_request', self.request_callback, 10)
        self.get_logger().info('Grasp node ready and listening on /grasp_request')

    def request_callback(self, msg: String):
        request = msg.data.strip()
        self.get_logger().info(f'Received grasp request: {request}')

        result = f'{request}_grasped'
        self.publisher.publish(String(data=result))
        self.get_logger().info(f'Published grasp result: {result}')


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
