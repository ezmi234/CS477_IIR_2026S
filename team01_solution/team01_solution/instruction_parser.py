"""
Instruction Parser Node

This node listens to the /instruction topic and processes incoming
string messages. For Phase 1, it's kept simple and just prints/logs
whatever instruction it receives so we can verify everything is wired up.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class InstructionParser(Node):
    """ROS node that listens for and handles incoming instruction messages."""

    def __init__(self):
        super().__init__('instruction_parser')

        # Set up subscription to receive instructions
        self.subscription = self.create_subscription(
            String,
            '/instruction',
            self.instruction_callback,
            10
        )

        self.get_logger().info("Instruction parser is running and listening on /instruction")

    def instruction_callback(self, msg: String):
        """Called every time a new instruction is received."""
        instruction = msg.data.strip()
        self.get_logger().info(f"Got instruction: {instruction}")


def main(args=None):
    """Starts the instruction parser node."""
    rclpy.init(args=args)

    node = InstructionParser()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()