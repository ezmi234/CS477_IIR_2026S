#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity
from ament_index_python.packages import get_package_share_directory
import manip_challenge.misc as misc
import time

# sdf_path = os.path.join(get_package_share_directory('manip_challenge'), 'models')



def spawn_sdf_object(node, client, object_name, xyzrpy, name=None):
    """ Spawn an object """
    
    # Set data for request
    sdf_file_path = os.path.join(
    get_package_share_directory("manip_challenge"), "data", "models",
    object_name, "model.sdf")
    request = SpawnEntity.Request()
    request.name = name or object_name
    request.xml = open(sdf_file_path, 'r').read()
    request.robot_namespace = object_name
    request.initial_pose = misc.list2Pose(xyzrpy)

    node.get_logger().info("Sending service request to `/spawn_entity`")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print('response: %r' % future.result())
    else:
        raise RuntimeError(
            'exception while calling service: %r' % future.exception())

def main():
    # Start node
    rclpy.init()
    
    node = rclpy.create_node("entity_spawner")
    time.sleep(5)

    node.get_logger().info(
        'Creating Service client to connect to `/spawn_entity`')
    client = node.create_client(SpawnEntity, "/spawn_entity")

    node.get_logger().info("Connecting to `/spawn_entity` service...")
    if not client.service_is_ready():
        client.wait_for_service()
        node.get_logger().info("...connected!")
    
    # Spawn object
    spawn_sdf_object(node, client, 'coke_can', [0.55,0.25,0.6, 0, 0, 0])
    spawn_sdf_object(node, client, 'strawberry', [0.5,-0.15,0.6, 0, 0, 0])
    spawn_sdf_object(node, client, 'meat_can', [0.5,0.0,0.6, 0, 0, 0])
    spawn_sdf_object(node, client, 'hammer', [0.5,0.15,0.6, 0, 0, np.pi/2])
    spawn_sdf_object(node, client, 'banana', [0.7,-0.2,0.6, 0, 0, -np.pi/2])

    node.get_logger().info("Done! Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
