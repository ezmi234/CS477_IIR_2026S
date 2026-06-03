#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringPose
from geometry_msgs.msg import Pose

class ObjectDetectClient(Node):
    """Client node that calls the object detection service with a text prompt."""

    def __init__(self):
        """Create the detection service client and wait until the service is available."""
        super().__init__('string_pose_client')
        # Service client
        self.cli = self.create_client(StringPose, 'detect_objects_with_prompt')
        
        # Waiting a service
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service is not available. Waiting...')
        
        self.req = StringPose.Request()

    def send_request(self, text):
        """Send the input text as a request and return the service response."""
        self.req.data = text
        
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    """Run the detection client node and send an example request."""
    rclpy.init()
    
    client = ObjectDetectClient()
    
    response = client.send_request("Detect a meat_can and return [ymin, xmin, ymax, xmax, label]")
    
    if response is not None:
        client.get_logger().info('Service call success!')
    else:
        client.get_logger().error('Service call failed!!')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
