# Manipulation challenge tutorial 1

In this tutorial, you will learn how to pick up an object from a tabletop scene with multiple objects spawned in the Gazebo simulator.

[!IMPORTANT]
Always source your workspace environment whenever you open a new terminal:
~~~~bash
source ./install/setup.bash
~~~~

[!IMPORTANT]
The repository has been updated. Pull the latest changes and build the platform packages:
~~~~bash
git pull
colcon build --symlink-install --packages-select ur5_ros2_gazebo
~~~~

## 1.0 Build Instructions
If you have not built this package yet, or if you have modified the code, run the following command from the workspace root:
~~~~bash
colcon build --symlink-install --packages-select manip_challenge
~~~~

## 1.1 Initiate the Pick-and-Place Environment

To launch the Gazebo simulation environment, open Terminal 1 and run:
~~~~bash
ros2 launch manip_challenge ur5_setup_set1_picking.launch.py
~~~~

**Configuration Steps**:
1. Select the `position` controller by pressing `1` when prompted. You will need to do this twice.
2. Select `GAZEBO (GUI)` mode by pressing `1`.
3. If this is your first time launching the environment, model download and loading may take a few minutes.

**Grasping an Object**
You can control the simulated UR5e arm and gripper by sending a sequence of joint commands. Open Terminal 2 and run:
~~~~bash
ros2 run manip_challenge example1_grasping
~~~~
Tip: Check `example1.py` to see how joint position commands are sent and how the gripper is opened and closed.


## 1.2 Access Ground-Truth Object Poses
Although the full challenge requires perception, you may use Gazebo's internal state for debugging and dry runs. To query the ground-truth poses of objects, run:
~~~~bash
ros2 run manip_challenge example2_gazebo_pose
~~~~

[!WARNING]
Subscribing to Gazebo internal topics for object states is strictly prohibited during the actual competition. This tool is provided for system testing and dry runs only.


## 1.3 Task Specification (Command System)
During the competition, the instructor or TA will broadcast task instructions through the `/task_commands` topic.

**Example Command Format**:
~~~~bash
d = "Move book, eraser, and soap to the left storage. Move snack, biscuits box, and soap to the right storage."
~~~~
Your robot system must subscribe to this topic and carry out the pick-and-place tasks accordingly. You can test the communication flow using the following scripts:

**Terminal 1 (Listener)**:
Use this to confirm that your system receives the command correctly.
~~~~bash
ros2 run manip_challenge example3_task_command_sub
~~~~

**Terminal 2 (Publisher)**:
Use this to simulate the instructor sending a command.
~~~~bash
ros2 run manip_challenge example3_task_command_pub
~~~~


# Manipulation Challenge Tutorial 2

In this tutorial, you will use the Gemini API to estimate the coordinates of a target object in a cluttered scene based on a text instruction.

Before proceeding, please review the [Gemini API quickstart](https://ai.google.dev/gemini-api/docs/quickstart).  
You must first create a project and generate an [API key](https://ai.google.dev/gemini-api/docs/api-key#api-keys).

You will also need to install the following packages:

~~~~bash
pip install -U google-genai Pillow
~~~~

## 2.1 Initialize the object pick-and-place environment
In the competition environment, about 10 random objects will appear on the table, and a text instruction will be delivered through a ROS 2 topic.

Start the random-object environment with:
~~~~bash
ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
~~~~

## 2.2 Detect an object using GEMINI AI studio
Set your Google AI Studio API key in the terminal:
~~~~bash
export GEMINI_API_KEY="your_api_key"
~~~~

Start the detection service that queries the Gemini model:
~~~~bash
ros2 run manip_challenge example4_detection_server
~~~~
Make sure to use your own Gemini API key obtained from https://aistudio.google.com.

Once the server is running, request the location of a target object by running the client:
~~~~bash
ros2 run manip_challenge example4_detection_client
~~~~

## 2.3 Grasp an object using GEMINI AI studio
Set your Google AI Studio API key in the terminal:
~~~~bash
export GEMINI_API_KEY="your_api_key"
~~~~

Start the grasping service that queries the Gemini model:
~~~~bash
ros2 run manip_challenge example5_grasp_server
~~~~

Once the server is running, request the location of a designated object by running the client:
~~~~bash
ros2 run manip_challenge example5_grasp_client "Detect a meat can and return [ymin, xmin, ymax, xmax, label]"
~~~~

# Competition
The goal of this challenge is to develop a robust robotic system capable of executing high-level instructions through any intelligence pipeline. Participants must implement a system that performs instruction subscription, object detection, and robot manipulation within a simulated environment.

Our examplar instruction is as follow:
~~~~bash
d = "Move the banana and the meat can to the left storage. Move the strawberry and the hammer in the right storage. Move the coke can on the shelf."
~~~~

Our competition process flow is as follow:
- Participants must launch their code initially and wait in a "Standby State." The system should only begin processing once it receives an instruction via a specific ROS topic.
~~~~bash
/task_commands
~~~~

- Each team is allocated a maximum of 8 minutes to complete all assigned tasks. If the tasks are not completed within this timeframe, the evaluation will be based on the number of successful manipulations achieved up to the 8-minute mark.

