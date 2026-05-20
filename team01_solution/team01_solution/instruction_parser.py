"""
Instruction Parser Node

This node listens to /instruction, parses a minimal structured command,
and forwards pipeline requests to downstream topics.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def parse_instruction_text(text: str):
    """Parse a generic instruction into action, object, and optional target."""
    if text is None:
        return None

    normalized = text.strip()
    if not normalized:
        return None

    if ':' in normalized:
        tokens = [token.strip() for token in normalized.split(':') if token.strip()]
    else:
        tokens = [token.strip() for token in normalized.split() if token.strip()]

    if len(tokens) < 2:
        return None

    action = tokens[0]
    object_name = tokens[1]
    target = ':'.join(tokens[2:]) if len(tokens) > 2 else None

    return {
        'action': action,
        'object': object_name,
        'target': target,
        'raw': normalized,
    }


class InstructionParser(Node):
    """ROS node that orchestrates instruction-driven pipeline requests."""

    def __init__(self):
        super().__init__('instruction_parser')

        self.detect_publisher = self.create_publisher(String, '/detect_request', 10)
        self.grasp_publisher = self.create_publisher(String, '/grasp_request', 10)
        self.motion_publisher = self.create_publisher(String, '/motion_request', 10)

        self.create_subscription(String, '/instruction', self.instruction_callback, 10)
        self.create_subscription(String, '/detect_result', self.detect_result_callback, 10)
        self.create_subscription(String, '/grasp_result', self.grasp_result_callback, 10)
        self.create_subscription(String, '/motion_result', self.motion_result_callback, 10)

        self.current_instruction = None

        self.get_logger().info('Instruction parser running and listening on /instruction')

    def instruction_callback(self, msg: String):
        instruction = msg.data
        parsed = parse_instruction_text(instruction)

        if parsed is None:
            self.get_logger().warn(
                f"Could not parse instruction '{instruction}'. "
                "Use formats like 'pick:banana' or 'place:cup:basket_a'"
            )
            return

        self.current_instruction = parsed
        self.get_logger().info(
            f"Parsed instruction: action={parsed['action']} "
            f"object={parsed['object']} target={parsed['target']}"
        )

        detect_request = parsed['object']
        self.detect_publisher.publish(String(data=detect_request))
        self.get_logger().info(f"Published /detect_request: {detect_request}")

    def detect_result_callback(self, msg: String):
        detected = msg.data.strip()
        self.get_logger().info(f"Received /detect_result: {detected}")

        if self.current_instruction is None:
            self.get_logger().warn('No active instruction to forward to grasp stage')
            return

        grasp_request = f"{self.current_instruction['action']}:{self.current_instruction['object']}"
        self.grasp_publisher.publish(String(data=grasp_request))
        self.get_logger().info(f"Published /grasp_request: {grasp_request}")

    def grasp_result_callback(self, msg: String):
        grasp_result = msg.data.strip()
        self.get_logger().info(f"Received /grasp_result: {grasp_result}")

        if self.current_instruction is None:
            self.get_logger().warn('No active instruction to forward to motion stage')
            return

        motion_request = f"{self.current_instruction['action']}:{self.current_instruction['object']}"
        self.motion_publisher.publish(String(data=motion_request))
        self.get_logger().info(f"Published /motion_request: {motion_request}")

    def motion_result_callback(self, msg: String):
        motion_result = msg.data.strip()
        self.get_logger().info(f"Received /motion_result: {motion_result}")
        self.get_logger().info('Pipeline completed for current instruction')


def main(args=None):
    """Starts the instruction parser node."""
    rclpy.init(args=args)

    node = InstructionParser()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
