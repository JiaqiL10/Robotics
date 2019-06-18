# Robotic Cluedo
#### YouTube link: <https://youtu.be/6qkZqz-cvWg>

The following is a final year project with aim of making a robot (in this case a Turtlebot) play a version of the board game, Cluedo. In our version, the robot has to search the environment (a room) trying to find the murderer and the weapon of the crime.

The tasks that the robot should perform are the following:

1. Localisation
2. Object search and detection
3. Object identification

## Commands

I have written this README with the following assumptions:

1. You are using an Ubuntu terminal.
2. You are cloning our Group4 git repository (as a folder called `src`) into an already set up `catkin_ws` folder, in the `home` directory of whichever user you are currently logged into.

### Run Gazebo
```
cd catkin_ws/
export TURTLEBOT_GAZEBO_WORLD_FILE=Cluedo.world
roslaunch turtlebot_gazebo turtlebot_world.launch
```
### Load the map
```
cd catkin_ws/
cd $HOME/catkin_ws/src/lab4/launch
roslaunch simulated_localisation.launch map_file:=$HOME/catkin_ws/src/lab4/maps/examplemap.yaml
```
### Run RViz
```
cd catkin_ws/
roslaunch turtlebot_rviz_launchers view_navigation.launch
```
### Load AR Tracking
```
cd $HOME/catkin_ws/src/lab5/launch
roslaunch ar_tracking.launch
```
## Executing Cluedo.py
If you have followed the commands so far, you will also need to let the robot know where it is in the map. To do this, there are two options:

1. Briefly run, then exit `turtlebot_teleop `
2. In RViz, use the button `2D Pose Estimate`

If you want to use `turtlebot_teleop`, then use this command in a new Ubuntu terminal window

```
roslaunch turtlebot_teleop keyboard_teleop.launch
```

You are also free to load up `ar_pose_marker` in a new Ubuntu terminal window with this command:

```
rostopic echo ar_pose_marker
```

Then finally, you are able to run our Group4.py file with these commands:

```
chmod u+x $HOME/catkin_ws/src/lab5/src/Cluedo.py
rosrun  Cluedo.py
```


## To run our code on a different map
This requires that new points are generated for another `.yaml` file.

To assign goal points on a different map (not the examplemap), you will need to run another piece of code in our `lab5/src/` directory. The file you need to edit is:

```
$HOME/catkin_ws/src/lab5/src/goal_points.py
```

Change the two variables `image` and `yaml`. I have pasted the code below for your reference.

```
image = read_pgm(home + "/catkin_ws/src/lab4/maps/examplemap.pgm", byteorder='<') # the map should be in the same
yaml = home +"/catkin_ws/src/lab4/maps/examplemap.yaml"
```

Then to run the `goal_points.py` use the following commands:

```
chmod u+x $HOME/catkin_ws/src/lab5/src/goal_points.py
rosrun lab5 goal_points.py
```
This will change the points.csv file, located at `/catkin_ws/src/lab5/`.

To re-run our `Cluedo.py`, use the same commands again, as above:

```
chmod u+x $HOME/catkin_ws/src/lab5/src/Group4.py
rosrun lab5 Group4.py
```
