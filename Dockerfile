FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    build-essential \
    python3-rosdep \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep update || true

# Create workspace
WORKDIR /root/cs477_ws
RUN mkdir -p /root/cs477_ws/src

# Copy only essential packages: team_1 and manip_challenge
COPY team_1 /root/cs477_ws/src/team_1
COPY manip_challenge /root/cs477_ws/src/manip_challenge

# Install ROS dependencies for only these packages
RUN rosdep install --from-paths /root/cs477_ws/src --ignore-src -r -y || true

# Build only the essential packages
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-up-to team_1 --symlink-install"

# Ensure shells source ROS and workspace on start
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /root/cs477_ws/install/setup.bash" >> ~/.bashrc

# Add entrypoint
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["bash"]
