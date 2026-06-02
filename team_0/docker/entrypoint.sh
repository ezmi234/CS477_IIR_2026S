#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /root/cs477_ws/install/local_setup.bash ]; then
  source /root/cs477_ws/install/local_setup.bash
fi

exec "$@"
