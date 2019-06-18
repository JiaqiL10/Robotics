#!/usr/bin/env python

import rospy
from kobuki_msgs.msg import BumperEvent
from geometry_msgs.msg import Twist

def callback(data):
    if (data.state == BumperEvent.PRESSED):       
        rospy.loginfo(rospy.get_caller_id() + 'Bumper state: %s', data.state)
        rospy.signal_shutdown("Hit an object")

def publisher():
	pub = rospy.Publisher('mobile_base/commands/velocity', Twist,queue_size=10)
        rospy.Subscriber('mobile_base/events/bumper',BumperEvent,callback)	
	rospy.init_node('Walker', anonymous=True)
	rate = rospy.Rate(10) #10hz
	desired_velocity = Twist()
	while not rospy.is_shutdown():		
	    desired_velocity.linear.x = 0.2 # Forward with 0.2 m/sec.
	    for i in range (30):
		    pub.publish(desired_velocity)
		    rate.sleep()
		
	    desired_velocity.linear.x = 0
	    for i in range (5):
		    pub.publish(desired_velocity)
		    rate.sleep()
		
	    desired_velocity.angular.z = 3.14159
	    for i in range(10):
		    pub.publish(desired_velocity)
		    rate.sleep()
		
	    desired_velocity.angular.z = 0
	    for i in range(5):
		    pub.publish(desired_velocity)
		    rate.sleep()

if __name__ == "__main__":
	try:
		publisher()
	except rospy.ROSInterruptException:
		pass
