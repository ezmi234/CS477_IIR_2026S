# Team 01 Solution

The previous separate prototype nodes have been integrated into the final
`team01_solution` runtime. See `README_team_1.md` for the final launch and
submission procedure.

Main command:

```bash
ros2 launch team01_solution contest_run.launch.py
```



```bash
cd ~/cs477_ws
rm -rf build/team01_solution install/team01_solution build/team_1 install/team_1 log

colcon build --symlink-install --packages-select team01_solution team_1 team01_solution_msgs
source install/setup.bash
```

Terminal 1:
```bash
ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
```

Terminal 2:
```bash
ros2 launch team_1 contest_run.launch.py
```

Terminal 3:
```bash
ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'Move the banana to the left storage.'}"
```