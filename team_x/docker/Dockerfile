FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONNOUSERSITE=1
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-lxml \
    python3-opencv \
    python3-pykdl \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

COPY apt-packages.txt /tmp/cs477_apt_packages.txt
COPY requirements.txt /tmp/cs477_requirements.txt
COPY ros-humble-gazebo-grasp-plugin_1.0.2-0jammy_amd64.deb /tmp/
COPY ros-humble-gazebo-version-helpers_0.0.0-0jammy_amd64.deb /tmp/
COPY ros-humble-roboticsgroup-upatras-gazebo-plugins_0.2.0-0jammy_amd64.deb /tmp/

RUN apt-get update && \
    xargs -a /tmp/cs477_apt_packages.txt apt-get install -y --no-install-recommends && \
    dpkg -i /tmp/ros-humble-gazebo-grasp-plugin_1.0.2-0jammy_amd64.deb \
      /tmp/ros-humble-gazebo-version-helpers_0.0.0-0jammy_amd64.deb \
      /tmp/ros-humble-roboticsgroup-upatras-gazebo-plugins_0.2.0-0jammy_amd64.deb && \
    rm -rf /var/lib/apt/lists/* /tmp/*.deb

RUN python3 -m pip install --no-cache-dir -U \
    -r /tmp/cs477_requirements.txt \
    "setuptools<66.0" \
    numpy==1.24.4 \
    scipy==1.10.1 \
    opencv-python==4.10.0.84 \
    ultralytics==8.4.60 \
    torch==2.3.1 \
    transformers==4.41.2 \
    google-genai \
    Pillow

WORKDIR /root/cs477_ws

COPY . /root/cs477_ws/src/cs477_IIR
COPY team_x/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN cp /root/cs477_ws/src/cs477_IIR/utils/include/move_group_interface_improved.h \
      /opt/ros/humble/include/moveit/move_group_interface/ && \
    source /opt/ros/humble/setup.bash && \
    vcs import --input src/cs477_IIR/docs/common_sim.repos --workers 1 src && \
    (rosdep init || true) && \
    rosdep update && \
    rosdep install -i --from-path src --rosdistro humble -y -r \
      --skip-keys "gazebo_grasp_plugin gazebo_version_helpers roboticsgroup_upatras_gazebo_plugins" && \
    colcon build --symlink-install \
      --packages-ignore gazebo_plugins realsense_gazebo_plugin gym-gazebo2 gazebo_ros \
      --allow-overriding gazebo_dev gazebo_msgs gazebo_ros \
      --cmake-args -DBUILD_TESTING=OFF && \
    colcon build --symlink-install \
      --packages-select gazebo_plugins realsense_gazebo_plugin \
      --packages-skip gazebo_ros \
      --allow-overriding gazebo_plugins \
      --parallel-workers=1 \
      --cmake-args -DCMAKE_CXX_FLAGS="--param ggc-min-expand=20" \
      --cmake-args -DBUILD_TESTING=OFF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "team_x", "contest_run.launch.py"]
