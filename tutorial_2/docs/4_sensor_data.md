# Practice 4: Sensor Data Acquisition

Realsense d435 camera is an RGB-D camera that produces RGB and depth images. You can find the topics:
~~~~bash
ros2 topic list | grep camera
~~~~

Then, you can subscribe the images and also visualize it through python or RViZ. Here are the examples.
~~~~bash
# Top-down view camera
ros2 topic echo /camera/camera/color/image_raw
# Wrist camera
ros2 topic echo /wrist_camera/wrist_camera/color/image_raw
~~~~
In RViZ, you can also click the add Image to visualize the image stream from the camera.


You can subscribe the topic and store it using opencv.
~~~~bash
ros2 run tutorial_2 5_camera_image_saver
~~~~
If you want object detection, yolu can finetune the MASK R-CNN. If you need python3, you can use virtual environment for it. Let me know, if you need example for the virtual environment with ROS. 


