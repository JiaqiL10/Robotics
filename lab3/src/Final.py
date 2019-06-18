#!/usr/bin/env python
# This final piece fo skeleton code will be centred around gettign the students to follow a colour and stop upon sight of another one.

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
		# Initialise some standard movement messages such as a simple move forward and a message with all zeroes (stop)s
		# Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use

		self.bridge = CvBridge()
		rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)

	def callback(self, data):
		# Convert the received image into a opencv image
		# But remember that you should always wrap a call to this conversion method in an exception handler
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		# Set the upper and lower bounds for the two colours you wish to identify

		hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
		hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

		hsv_red_lower = np.array([10 - self.sensitivity, 100, 100])
		hsv_red_upper = np.array([5 + self.sensitivity, 255, 255])

		hsv_blue_lower = np.array([110 - self.sensitivity, 100, 100])
		hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

		# Convert the rgb image into a hsv image
		hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

		maskg = cv2.inRange(hsv, hsv_green_lower, hsv_green_upper)
		maskr = cv2.inRange(hsv, hsv_red_lower, hsv_red_upper)
		maskb = cv2.inRange(hsv, hsv_blue_lower, hsv_blue_upper)

		maskForRed = cv2.bitwise_and(cv_image, cv_image, mask=maskr)
		red_and_green = cv2.bitwise_or(maskg, maskr)
		all_colors = cv2.bitwise_or(red_and_green, maskb)

		kernelOpen = np.ones((5, 5))
		kernelClose = np.ones((20, 20))

		maskOpen = cv2.morphologyEx(all_colors, cv2.MORPH_OPEN, kernelOpen)
		maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)


		# find edges
		edged = cv2.Canny(all_colors, 50, 200)


		try:
			contours, hierarchy = cv2.findContours(maskClose.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		except Exception as e:
			print(e)



		hue_of_color_to_track = 60

		if len(contours) > 0:
			sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
			max_area = cv2.contourArea(sorted_contours[0])
			print(str(max_area))

			largest_area_center_point = cv2.moments(sorted_contours[0])
			cx = int(largest_area_center_point['m10'] / largest_area_center_point['m00'])
			cy = int(largest_area_center_point['m01'] / largest_area_center_point['m00'])

			height, width = cv_image.shape[:2]

			color_at_largest_area_center = cv_image[cy, cx]

			B = color_at_largest_area_center[0]
			G = color_at_largest_area_center[1]
			R = color_at_largest_area_center[2]

			rect = cv2.boundingRect(sorted_contours[0])
			x, y, w, h = rect

			if G > 0:
				self.blue_detected = 0
				self.red_detected = 0
				self.green_detected = 1
				print("Color is red")
			elif R > 0:
				self.blue_detected = 0
				self.red_detected = 1
				self.green_detected = 0
				print("Color is green")
			elif B > 0:
				self.blue_detected = 1
				self.red_detected = 0
				self.green_detected = 0
				print("Color is Blue")
			move = Twist()
			# Check if a flag has been set for the stop message
			if self.green_detected == 1:
				if max_area > 35000:
					# Too close to object, need to move backwards
					# linear = positive
					# angular = radius of minimum enclosing circle
					err = cx - (width / 2)
					move.linear.x = -0.1
					move.angular.z = -float(err) / 100
				elif max_area < 30000:
					# Too far away from object, need to move forwards
					# linear = positive
					# angular = radius of minimum enclosing circle
					err = cx - (width / 2)
					move.linear.x = 0.1
					move.angular.z = -float(err) / 100
					print(str(move))
				self.pub.publish(move)
			# self.<publisher_name>.publish(<Move>)
			else:
				self.pub.publish(Twist())

			# Be sure to do this for the other colour as well
			# Show the resultant images you have created. You can show all of them or just the end result if you wish to.
			cv2.drawContours(cv_image.copy(), contours, -1, (0, 255, 0), 3)
			# cv2.drawContours(cv_image, sorted_contours[0], -1, (255, 0, 0), 3)
			cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
			# cv2.circle(cv_image, (cx, cy), 10, (0, 0, 0), -1)
			cv2.imshow('Circle', cv_image)
			cv2.waitKey(3)
		else:
			self.pub.publish(Twist())
			print("no objects detected")


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


