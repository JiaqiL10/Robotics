#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import rospy
import sys

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class colourIdentifier():

    def __init__(self):
        # Initialise a publisher to publish messages to the robot base
        self.pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=10)
        # We covered which topic receives messages that move the robot in the 2nd Lab Session

        # Initialise any flags that signal a colour has been detected in view
        self.red_detected = 0
        self.blue_detected = 0
        self.green_detected = 0

        # Initialise the value you wish to use for sensitivity in the colour detection (10 should be enough)
        self.sensitivity = 10
        # Initialise some standard movement messages such as a simple move forward and a message with all zeroes (stop)
        self.move_forward = Twist()
        self.move_forward.linear.x = 0.2

        self.stop = Twist()
        self.stop.linear.x = 0
        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use

        self.bridge = CvBridge()
        image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)

    # We covered which topic to subscribe to should you wish to receive image data

    def callback(self, data):
        # Convert the received image into a opencv image
        # But remember that you should always wrap a call to this conversion method in an exception handler

        # Set the upper and lower bounds for the two colours you wish to identify
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

        hsv_red_lower = np.array([10 - self.sensitivity, 100, 100])
        hsv_red_upper = np.array([5 + self.sensitivity, 255, 255])

        hsv_blue_lower = np.array([110 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        maskg = cv2.inRange(hsv, hsv_green_lower, hsv_green_upper)
        maskr = cv2.inRange(hsv, hsv_red_lower, hsv_red_upper)
        maskb = cv2.inRange(hsv, hsv_blue_lower, hsv_blue_upper)

        # Convert the rgb image into a hsv image

        # Filter out everything but particular colours using the cv2.inRange() method

        # To combine the masks you should use the cv2.bitwise_or() method
        # You can only bitwise_or two image at once, so multiple calls are necessary for more than two colours

        # Apply the mask to the original image using the cv2.bitwise_and() method
        # As mentioned on the worksheet the best way to do this is to bitwise and an image with itself and pass the mask to the mask parameter
        # As opposed to performing a bitwise_and on the mask and the image.

        # Find the contours that appear within the certain colours mask using the cv2.findContours() method
        # For <mode> use cv2.RETR_LIST for <method> use cv2.CHAIN_APPROX_SIMPLE
        redand_green = cv2.bitwise_or(maskg, maskr)
        all_colors = cv2.bitwise_or(redand_green, maskb)

        edged = cv2.Canny(all_colors, 50, 200)
        # cv2.imshow('Canny Edges', edged)

        try:
            contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cv_image, contours, -1, (0, 255, 0), 3)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cv2.drawContours(cv_image, sorted_contours[0], -1, (255, 0, 0), 3)
        except Exception as e:
            print(e)

        # get center of largest contour area
        M = cv2.moments(sorted_contours[0])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        image_at_center = cv_image[cy, cx]
        B = image_at_center[0]
        G = image_at_center[1]
        R =image_at_center[2]

        if B > 0:
            self.blue_detected = 1
            print("Color is blue")
        elif G > 0:
            self.green_detected = 1
            print("Color is green")
        elif R > 0:
            self.red_detected = 1
            print("Color is red")

        cv2.circle(cv_image, (cx, cy), 10, (0, 0, 0), -1)
        cv2.imshow('Circle ', cv_image)

        # Loop over the contours
        # There are a few different methods for identifying which contour is the biggest
        # Loop throguht the list and keep track of whioch contour is biggest or
        # Use the max() method to find the largest contour
        # M = cv2.moments(c)
        # cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

        # Check if the area of the shape you want is big enough to be considered
        # If it is then change the flag for that colour to be True(1)
        # if colour_max_area > #<What do you think is a suitable area?>:
        # draw a circle on the contour you're identifying as a blue object as well
        # cv2.circle(<image>,(<center x>,<center y>),<radius>,<colour (rgb tuple)>,<thickness (defaults to 1)>)
        # Then alter the values of any flags

        # Show the resultant images you have created. You can show all of them or just the end result if you wish to.
        cv2.waitKey(3)


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
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()


# Check if the node is executing in the main path
if __name__ == '__main__':
    main(sys.argv)



