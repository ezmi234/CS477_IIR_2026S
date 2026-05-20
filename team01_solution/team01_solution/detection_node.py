"""
Detection node placeholder.

Subscribes to /detect_request and publishes a dummy /detect_result.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DetectionNode(Node):

    def __init__(self):
        super().__init__('detection_node')
        self.publisher = self.create_publisher(String, '/detect_result', 10)
        self.create_subscription(String, '/detect_request', self.request_callback, 10)
        self.get_logger().info('Detection node ready and listening on /detect_request')

    def request_callback(self, msg: String):
        request = msg.data.strip()
        self.get_logger().info(f'Received detect request: {request}')

        result = f'{request}_detected'
        self.publisher.publish(String(data=result))
        self.get_logger().info(f'Published detect result: {result}')


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
