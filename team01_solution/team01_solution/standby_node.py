import rclpy
from rclpy.node import Node


class StandbyNode(Node):

    def __init__(self):
        super().__init__('standby_node')

        self.get_logger().info("System is in standby mode.")


def main(args=None):

    rclpy.init(args=args)

    node = StandbyNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()