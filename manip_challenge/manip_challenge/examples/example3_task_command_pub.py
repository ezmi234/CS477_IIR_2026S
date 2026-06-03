#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String


def send_task_command(node):
    """Send a string command """
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
        
    publisher = node.create_publisher(String, '/task_commands', qos)

    msg = String()
    msg.data = "Move a book, an eraser, and a soap to the left storage. Move a snack, a biscuits box, and a soap to the right storage."
    publisher.publish(msg)
    node.get_logger().info("task_command_pub: published!")
    
        
def main():
    rclpy.init()

    node = rclpy.create_node("task_command_pub")    
    rclpy.spin_once(node, timeout_sec=1)
    node.get_logger().info("task_command_pub: publish task commands")

    # Send a list of commands
    send_task_command(node)
    
    node.destroy_node()
    rclpy.shutdown()        
    
if __name__ == '__main__':
    main()
