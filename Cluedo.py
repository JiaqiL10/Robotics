#!/usr/bin/env python
# This final piece fo skeleton code will be centred around gettign the students to follow a colour and stop upon sight of another one.

from __future__ import division
import cv2
import math, numpy
import rospy
import tf
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from  geometry_msgs . msg  import  PoseWithCovarianceStamped , Twist
from  geometry_msgs . msg  import  TwistWithCovarianceStamped , Twist
import roslib
from ar_track_alvar_msgs.msg import AlvarMarkers
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, Vector3, Point, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
import actionlib
from actionlib_msgs.msg import *
from std_msgs.msg import Bool
from math import radians
#import firstwalk
import random
import time
import csv
import re
import os
from os.path import expanduser
home = expanduser("~")
f = open(home +"/catkin_ws/src/lab5/points.csv")
csv_f = list(csv.reader(f))
#### their should be 18 points chnage to make it just that much
### read each line , CSV module how to read parse and write.
#linex1 = re.sub("[()!@#$]", " ", csv_f[0][0])

linex1 = re.sub('[()]', '', csv_f[0][0]).strip()
print float(linex1)
liney1 = re.sub('[()]', '', csv_f[0][1]).strip()#4.3
print float(liney1)
linex3 = re.sub("[()]", " ", csv_f[1][0])#-5.65
print float (linex3)
liney3 = re.sub("[()]", " ", csv_f[1][1])#4.25
print float (liney3)
linex5 = re.sub("[()!@#$]", " ", csv_f[3][0])#-4.5
print float (linex5)
liney5 = re.sub("[()!@#$]", " ", csv_f[3][1])#4.4
print float (liney5)
linex7 = re.sub("[()!@#$]", " ", csv_f[4][0])#-3.35
print float (linex7)
liney7 = re.sub("[()!@#$]", " ", csv_f[4][1])#4.35
print float (liney7)
linex9 = re.sub("[()!@#$]", " ", csv_f[6][0])#-2.2
print float (linex9)
liney9 = re.sub("[()!@#$]", " ", csv_f[6][1])#4.45
print float (liney9)
linex8 = re.sub("[()!@#$]", " ", csv_f[7][0])#-3.35
print float (linex8)
liney8 = re.sub("[()!@#$]", " ", csv_f[7][1])#-1.7763568394002505e-15
print float (liney8)
linex6 = re.sub("[()!@#$]", " ", csv_f[9][0])#-4.5
print float (linex6)
liney6 = re.sub("[()!@#$]", " ", csv_f[9][1])#0.05
print float (liney6)
linex4 = re.sub("[()!@#$]", " ", csv_f[10][0])#-5.65
print float (linex4)
liney4 = re.sub("[()!@#$]", " ", csv_f[10][1])#0.15
print float (liney4)
linex2 = re.sub("[()!@#$]", " ", csv_f[12][0])#-6.8
print float (linex2)
liney2 = re.sub("[()!@#$]", " ", csv_f[12][1])#0.05
print float (liney2)
#############
linex19 = re.sub('[()]', '', csv_f[5][0]).strip()
print float(linex19)
liney19 = re.sub('[()]', '', csv_f[5][1]).strip()#4.3
print float(liney19)
linex10 = re.sub('[()]', '', csv_f[13][0]).strip()
print float(linex10)
liney10 = re.sub('[()]', '', csv_f[13][1]).strip()#4.3
print float(liney10)
linex11 = re.sub("[()]", " ", csv_f[15][0])#-5.65
print float (linex11)
liney11 = re.sub("[()]", " ", csv_f[15][1])#4.25
print float (liney11)
linex12 = re.sub("[()!@#$]", " ", csv_f[16][0])#-4.5
print float (linex12)
liney12 = re.sub("[()!@#$]", " ", csv_f[16][1])#4.4
print float (liney12)
linex13 = re.sub("[()!@#$]", " ", csv_f[18][0])#-3.35
print float (linex13)
liney13 = re.sub("[()!@#$]", " ", csv_f[18][1])#4.35
print float (liney13)
linex14 = re.sub("[()!@#$]", " ", csv_f[19][0])#-2.2
print float (linex14)
liney14 = re.sub("[()!@#$]", " ", csv_f[19][1])#4.45
print float (liney14)
linex15 = re.sub("[()!@#$]", " ", csv_f[21][0])#-3.35
print float (linex15)
liney15 = re.sub("[()!@#$]", " ", csv_f[21][1])#-1.7763568394002505e-15
print float (liney15)
linex16 = re.sub("[()!@#$]", " ", csv_f[22][0])#-4.5
print float (linex16)
liney16 = re.sub("[()!@#$]", " ", csv_f[22][1])#0.05
print float (liney16)
linex17 = re.sub("[()!@#$]", " ", csv_f[23][0])#-5.65
print float (linex17)
liney17 = re.sub("[()!@#$]", " ", csv_f[23][1])#0.15
print float (liney17)
linex18 = re.sub("[()!@#$]", " ", csv_f[11][0])#-6.8
print float (linex18)
liney18 = re.sub("[()!@#$]", " ", csv_f[11][1])#0.05
print float (liney18)
linex20 = re.sub("[()!@#$]", " ", csv_f[2][0])#-6.8
print float (linex18)
liney20 = re.sub("[()!@#$]", " ", csv_f[2][1])#0.05
print float (liney18)

class colourIdentifier():

    def __init__(self):
        # these are the global variables i am using for tracking the states of various things the robot
        # is doing
        self.have_reached_center = False
        self.moving_to_goal = False
        self.found_marker_1 = False
        self.found_marker_2 = False
        self.marker_found_counter = 0
        self.target_index = 0
        self.marker_points = []
        self.correct_marker_points = []
        self.existing_marker_counter = 0
        self.last_saved_image = ''

        # these are topics i am using globally
        self.pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=10)
        self.img_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.marker_reached)
        self.img_sub.unregister()
        self._pose = Pose()
        self.poses_dict = {"pose":self._pose}
        self._twist = Twist()
        self.poses_dict = {"twist":self._twist}
        self.listener = tf.TransformListener()
        self._pose_sub = rospy.Subscriber('/odom', Odometry , self.sub_callback)
        self.initialPose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)
        self.get_initial_pose()
        # these are global instan

        # these are global instances of classes i am using
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.move = Twist()
        self.rate = rospy.Rate(10)

        # these are topics i am listening to
        rospy.Subscriber("center_has_been_reached", Bool, self.arrived_at_center)
        rospy.Subscriber("move_to_next_point", Bool, self.move_to_targets)
        rospy.Subscriber("marker_detected", AlvarMarkers, self.find_marker_in_map)
        rospy.Subscriber("ar_pose_marker", AlvarMarkers, self.ar_marker_detected)

        # What to do if shut down (e.g. Ctrl-C or failure)
        rospy.on_shutdown(self.shutdown)

        # Tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        # Allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))
        self.get_initial_pose()

    def sub_callback(self, msg):

        self._oriy = msg.pose.pose.orientation.y
        self._oriz = msg.pose.pose.orientation.z
        self._posew = msg.pose.pose.orientation.w
        self._twistx = msg.twist.twist.linear.x
        self._twisty = msg.twist.twist.linear.y
        self._twistz = msg.twist.twist.angular.z
    def get_initial_pose(self):
        time.sleep(2)
        self.poses_dict["pose"] = self._oriy, self._oriz, self._posew
        rospy.loginfo("Written posex")
        time.sleep(2)
        self.poses_dict["twist"] = self._twistz, self._twisty, self._twistx
        rospy.loginfo("Written twistz")


        with open('poses.txt', 'w') as file:

            for key, value in self.poses_dict.iteritems():
                if value:
                    file.write(str(key) + ':\n----------\n' + str(value) + '\n===========\n')

        rospy.loginfo("auto generate pose")
    #	rospy.init_node('check_odometry')
    		#odom_sub = rospy.Subscriber('/odom', Odometry, self.sub_callback)
    		#self.goal_sent = False
    	initial_pose = PoseWithCovarianceStamped()
        initial_twist = TwistWithCovarianceStamped()
        initialtrans = (trans,rot) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))

    	initialPose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)
    	initial_pose.header.stamp = rospy.Time.now()
    	initial_pose.header.frame_id = "map"
    	initial_pose.pose.pose.position.x = trans[0]#self._posex ##chnage these to be based on baselink
    	initial_pose.pose.pose.position.y = trans[1]#self._posey ##chnage these to be based on baskelink
        initial_pose.pose.pose.orientation.y = self._oriy
        initial_pose.pose.pose.orientation.z = self._oriz
    	initial_pose.pose.pose.orientation.w = self._posew ##dont chnage this only change the x and y
        initial_twist.twist.twist.linear.x = self._twistx
        initial_twist.twist.twist.linear.y = self._twisty
        initial_twist.twist.twist.angular.z = self._twistz
    	initial_pose.pose.covariance[0] = 0.25
    	initial_pose.pose.covariance[7] = 0.25
    	initial_pose.pose.covariance[35] = 0.06
        rate = rospy.Rate(10)
    	i = 1
    	while i < 8:
    		initialPose_pub.publish(initial_pose)
    		i += 1
    		rate.sleep()

    def move_to_center(self):
        # Send a goal
        if not self.have_reached_center:
            self.moving_to_goal = True
            position = {'x': -3.05, 'y': 3.07}
            #position = {'x': -6.799999999999999, 'y': 0.049999999999998934}
            quaternion = {'r1': 0.000, 'r2': 0.000, 'r3': 0.4, 'r4': 0.41}

            rospy.loginfo("Go to (%s, %s) pose", position['x'], position['y'])

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose = Pose(Point(position['x'], position['y'], 0.000),
                                         Quaternion(quaternion['r1'], quaternion['r2'], quaternion['r3'],
                                                    quaternion['r4']))

            # Start moving
            self.move_base.send_goal(goal)

            # Allow TurtleBot up to 60 seconds to complete task
            success = self.move_base.wait_for_result(rospy.Duration(60))

            state = self.move_base.get_state()
            result = False

            if success and state == GoalStatus.SUCCEEDED:
                # We made it!
                result = True
            else:
                self.move_base.cancel_goal()

            self.moving_to_goal = False
            return result
        else:
            return True

    def arrived_at_center(self, msg):
        if msg.data and not self.have_reached_center:
            # set the flag to show that center has been reached
            # and we are not currently moving to any goal
            self.have_reached_center = True
            self.moving_to_goal = False

            # next step is to see if we detected and saved any marker points
            # on our way to center
            if len(self.marker_points) > 0:
                # we are moving to the first position in our list of possible marker points
                self.move_to_targets(self.target_index)
            else:
                # incase we did not detect any markers on our way here , let us being our search
                self.search_map()
        else:
            return

    def move_to_targets(self, index):
        try:
            # move to the target
            self.moving_to_goal = True
            rospy.loginfo("moving to (%s, %s,%s) pose", self.marker_points[index]['x'], self.marker_points[index]['y'],
                          self.marker_points[index]['theta'])

            goal = MoveBaseGoal()

            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()

            goal.target_pose.pose = Pose(Point(self.marker_points[index]['x'], self.marker_points[index]['y'], 0.000),
                                         Quaternion(0.000, 0.000, numpy.sin(self.marker_points[index]['theta'] / 2.0),
                                                    numpy.cos(self.marker_points[index]['theta'] / 2.0)))

            self.move_base.send_goal(goal)
            success = self.move_base.wait_for_result(rospy.Duration(60))
            state = self.move_base.get_state()

            if success and state == GoalStatus.SUCCEEDED:
                # We made it to the goal
                self.move_base.cancel_goal()
                self.moving_to_goal = False
                # next we are attempting to center ourself with the image
                # by listening to the transformation and adjusting our robot accordingly
                while True:
                    newPos = self.listener.lookupTransform('/base_link', '/ar_marker_0', rospy.Time(0))[0]
                    if newPos[1] > 0.02:
                        self.move.angular.z = 0.1
                    elif newPos[1] < -0.01:
                        self.move.angular.z = -0.1
                    else:
                        self.move.angular.z = 0
                        break
                    self.pub.publish(self.move)
                    self.rate.sleep()
                # once we have centered ourself properly,we want to basically open the camera feed
                self.start_camera()
            else:
                # incase we fail to reach the goal
                # cancel this current command
                self.move_base.cancel_goal()
                self.moving_to_goal = False

                # if we are not at the end of the list then move to the next point on the list of markers
                if self.target_index < (len(self.marker_points)):
                    print("More Markers")
                    self.target_index += 1
                    self.move_to_targets(self.target_index)
                else:
                    print("No More Markers")
                    self.rotate()
        except IndexError as e:
            print("No more points")

    def ar_marker_detected(self, data):
        number_of_markers = len(data.markers)
        marker_detected_pub = rospy.Publisher("marker_detected", AlvarMarkers, queue_size=10)
        if number_of_markers > 0:
            marker_detected_pub.publish(data)
            time.sleep(3)

    def find_marker_in_map(self, data):
        # if we have already found both markers then stop and return
        if self.found_marker_1 and self.found_marker_2:
            return
        else:
            # for each data marker sent
            for tag in data.markers:
                # calculate possible position of marker
                try:
                    mapArPos, rot = self.listener.lookupTransform('/map', '/ar_marker_0', rospy.Time(0))

                    rotationMatrix = self.listener.fromTranslationRotation(mapArPos, rot)

                    arZvector = numpy.array([rotationMatrix[0][2], rotationMatrix[1][2]])

                    newPosition = mapArPos[:2] + arZvector * 0.45

                    arZvectorInv = arZvector * (-1)

                    theta = math.atan2(arZvectorInv[1], arZvectorInv[0])
                    # first check if we have visited the marker
                    # if we have visited it then we move to the next marker
                    result = self.check_if_visited(newPosition[0],newPosition[1])
                    if result:
                        continue
                    else:
                        print("We have points in marker", len(self.marker_points))
                        self.marker_points.append({'x': newPosition[0], 'y': newPosition[1],'theta':theta})

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print("No AR Marker at point")

            if self.have_reached_center:
                self.moving_to_goal = True
                self.move_to_targets(self.target_index)

    def check_if_visited(self, x,y):
        # check if we have any points in the list
        if len(self.marker_points) > 0:
            # for each point in our list
            for point in self.marker_points:
                # calculate the difference between the points passed in and every point in the markers_list
                # if the difference is greater that 0.45 in any axis then it is likely a new point
                # so set the marker to 1 and if it is not set the marker back to 0
                x_axis_diff = abs(abs(point['x']) - abs(x))
                y_axis_diff = abs(abs(point['y']) - abs(y))

                print(x_axis_diff)
                print(y_axis_diff)

                if x_axis_diff < 0.25 and y_axis_diff < 0.25:
                    # its an old point
                    self.existing_marker_counter = self.existing_marker_counter + 1
                elif x_axis_diff > 0.25 and y_axis_diff > 0.25:
                    # its a new point
                    continue

            if self.existing_marker_counter > 0:
                # result if point is visited
                self.existing_marker_counter = 0
                return True
            else:
                # result if point is new
                self.existing_marker_counter = 0
                print("new point")
                return False
        else:
            # we dont have any points in the list so this is a new point
            return False

    def marker_reached(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Our threshold to indicate object detection
            threshold = 5
            # Load our image template, this is our reference image
            # Image Path
            img_src_folder = home + "/catkin_ws/src/lab5/img/"
            image_names = ["mustard.png", "plum.png", "rope.png", "scarlet.png", "revolver.png", "peacock.png",
                           "wrench.png"]

            list_of_matches = []

            # Get number of ORB matches
            for image in image_names:
                image_template = cv2.imread(img_src_folder + image, 0)
                matches = self.ORB_detector(cv_image, image_template)
                list_of_matches.append(matches)

            # get the highest image matches
            highest_match = max(list_of_matches)
            print(highest_match)
            if highest_match > threshold:
                #use this to get the name of the identified image from the list
                index = list_of_matches.index(highest_match)
                name = image_names[index]

                if self.last_saved_image == name:
                    return

                # do this to get contours
                imgray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(imgray, 30, 200)

                contours, heir = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                rect = cv2.boundingRect(sorted_contours[0])
                x, y, w, h = rect
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(cv_image, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                self.correct_marker_points.append(self.marker_points[self.target_index])
                self.save_image(cv_image, name)

            if self.target_index < (len(self.marker_points) - 1):
                self.target_index += 1
                self.move_to_targets(self.target_index)
            else:
                print("No More Markers")
                #self.search_map()
        except CvBridgeError as e:
            print(e)

    def search_map(self):
        if not self.found_marker_1 or not self.found_marker_2:
            if len(waypoints) > 0:
                client = actionlib.SimpleActionClient('move_base', MoveBaseAction)  # <3>
                client.wait_for_server()
                #while True: #### i might need to put a while about the ar_marker
                #while not self.found_marker_1: # only while a armarker hasnt been found
                for pose in waypoints:  # <4>
                    goal = goal_pose(pose)
                    client.send_goal(goal)
                    client.wait_for_result()
                    print (pose in waypoints)
                    print pose
                self.move_base.send_goal(goal)
                success = self.move_base.wait_for_result(rospy.Duration(60))

                state = self.move_base.get_state()
                if success and state == GoalStatus.SUCCEEDED:
                    print('blah')# We made it!
                    self.rotate()
                else:
                    self.move_base.cancel_goal()
                    self.moving_to_goal = False
                    #self.search_map()
            else:
                print("All points visited")
        else:
            print("All markers found")

    def rotate(self):
        global goToOnFlag
        goToOnFlag = False

        cmd_vel = rospy.Publisher("cmd_vel_mux/input/navi", Twist, queue_size=10)

        # create a Twist varaiable
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.linear.y = 0
        turn_cmd.linear.z = 0
        turn_cmd.angular.x = 0
        turn_cmd.angular.y = 0
        turn_cmd.angular.z = radians(20)

        t0 = rospy.Time.now().to_sec()

        current_angle = 0
        relative_angle = radians(360)

        while current_angle < relative_angle:
            cmd_vel.publish(turn_cmd)
            t1 = rospy.Time.now().to_sec()
            current_angle = radians(15) * (t1 - t0)

        turn_cmd.angular.z = 0
        cmd_vel.publish(turn_cmd)
        #self.search_map()

    def ORB_detector(self, new_image, image_template):
        # Function that compares input image to template
        # It then returns the number of ORB matches between them

        image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
        orb = cv2.ORB(1000, 1.2)

        # Detect keypoints of original image
        (kp1, des1) = orb.detectAndCompute(image1, None)

        # Detect keypoints of rotated image
        (kp2, des2) = orb.detectAndCompute(image_template, None)

        # Create matcher
        bf = cv2.BFMatcher()

        # Do matching
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        return len(good)

    def shutdown(self):
        if self.moving_to_goal:
            self.move_base.cancel_goal()
        rospy.loginfo("Stop")
        rospy.sleep(1)

    def start_camera(self):
        print("Camera Open")
        # subscribe to the camera topic
        self.img_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.marker_reached)

    def stop_camera(self):
        print("Closing camera")
        self.img_sub.unregister()

    def save_image(self, data,name):
        if self.found_marker_1 and self.found_marker_2:
            print("Already saved 2 markers")
            return
        elif not self.found_marker_1 and not self.found_marker_2:
            print("Found first marker")
            self.found_marker_1 = True
            self.last_saved_image = name
            path = home + "/catkin_ws/src/lab5/detections/"
            cv2.imwrite(os.path.join(path + name), data)
            self.stop_camera()
            return
        elif self.found_marker_1 and not self.found_marker_2:
            print("Found second marker")
            self.found_marker_2 = True
            self.last_saved_image = name
            path = home + "/catkin_ws/src/lab5/detections/"
            cv2.imwrite(os.path.join(path + name), data)
            self.stop_camera()
            return

waypoints = [
    [(float(linex1), float(liney1)), (0.0, 0.0, 0.0, 1.0)],
    [(float(linex3), float(liney3)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex5), float (liney5)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex7), float (liney7)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex9), float (liney9)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex8),float (liney8)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex6), float (liney6)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex4), float (liney4)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex2), float (liney2)), (0.0, 0.0, 0.0, 1.0)],
    [(float(linex10), float(liney10)), (0.0, 0.0, 0.0, 1.0)],
    [(float(linex11), float(liney11)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex12), float (liney12)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex13), float (liney13)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex14), float (liney14)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex15),float (liney15)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex16), float (liney16)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex17), float (liney17)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex18), float (liney18)), (0.0, 0.0, 0.0, 1.0)],
    [(float (linex19), float (liney19)), (0.0, 0.0, 0.0, 1.0)],
    #[(-6.49, 0.366), (0.0, 0.0, -0.984047240305, 0.177907360295)],
    [(float (linex20), float (liney20)), (0.0, 0.0, 0.0, 1.0)],
]
def goal_pose(pose):  # <2>
    goal_pose = MoveBaseGoal()
    goal_pose.target_pose.header.frame_id = 'map'
    goal_pose.target_pose.pose.position.x = pose[0][0]
    goal_pose.target_pose.pose.position.y = pose[0][1]
    goal_pose.target_pose.pose.position.z = 0.0
    goal_pose.target_pose.pose.orientation.x = pose[1][0]
    goal_pose.target_pose.pose.orientation.y = pose[1][1]
    goal_pose.target_pose.pose.orientation.z = pose[1][2]
    goal_pose.target_pose.pose.orientation.w = pose[1][3]

    return goal_pose
# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main(args):
    # Instantiate your class
    # And rospy.init the entire node
    rospy.init_node('colourIdentifier', anonymous=True)
    cI = colourIdentifier()
    # Ensure that the node continues running with rospy.spin()
    # You may need to wrap rospy.spin() in an exception handler in case of KeyboardInterrupts
    try:
        # Customize the following values so they are appropriate for your location
        success = cI.move_to_center()

        if success:
            goal_pub = rospy.Publisher("center_has_been_reached", Bool, queue_size=10)
            center_reached = True
            rate = rospy.Rate(10)
            i = 1
            while True:
                goal_pub.publish(center_reached)
                i += 1
                rate.sleep()
        else:
            print("The base failed to reach the desired pose")
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()


# Check if the node is executing in the main path
if __name__ == '__main__':
    main(sys.argv)
