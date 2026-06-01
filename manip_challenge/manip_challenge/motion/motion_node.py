#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from riro_srvs.srv import StringPose

from assignment_2.move_joint import ArmClient
from manip_challenge import move_gripper


TARGETS = {
    "storage_box_a": (0.55, -0.25, 0.18),
    "storage_box_b": (0.55, 0.25, 0.18),
    "storage_left": (0.55, -0.25, 0.18),
    "storage_right": (0.55, 0.25, 0.18),
    "bookshelf": (0.70, 0.00, 0.35),
}


def make_pose(x, y, z):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = 0.7071
    pose.orientation.y = 0.7071
    pose.orientation.z = 0.0
    pose.orientation.w = 0.0
    return pose


def above_pose(pose, dz):
    return make_pose(pose.position.x, pose.position.y, pose.position.z + dz)


class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")
        self.arm = ArmClient()
        self.create_service(StringPose, "motion_pick_place", self._pick_place_cb)

    def _get_target_pose(self, target_name):
        coords = TARGETS.get(target_name)
        if coords is None:
            return None
        return make_pose(*coords)

    def _pick_place_cb(self, request, response):
        target_pose = self._get_target_pose(request.data)
        if target_pose is None:
            self.get_logger().error(f"unknown target: {request.data}")
            return response

        object_pose = request.pose
        sequence = [
            above_pose(object_pose, 0.12),
            above_pose(object_pose, 0.01),
            above_pose(object_pose, 0.15),
            above_pose(target_pose, 0.12),
            target_pose,
            above_pose(target_pose, 0.15),
        ]

        move_gripper.gripper_open(self)
        self.arm.move_position(sequence[0], duration=2.0)
        self.arm.move_position(sequence[1], duration=2.0)
        move_gripper.gripper_close(self)
        self.arm.move_position(sequence[2], duration=2.0)
        self.arm.move_position(sequence[3], duration=2.0)
        self.arm.move_position(sequence[4], duration=2.0)
        move_gripper.gripper_open(self)
        self.arm.move_position(sequence[5], duration=2.0)

        response.pose = sequence[-1]
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
