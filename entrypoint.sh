#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
if [ -f /root/cs477_ws/install/setup.bash ]; then
  source /root/cs477_ws/install/setup.bash
fi

exec "$@"
