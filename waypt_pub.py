# Joseph Sommer and Yash Mhaskar

from __future__ import print_function
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import math
import time

# Global Variables
global state
state = 1
global reached
reached = 0

class WayptPub(Node):
	def __init__(self):
		super().__init__('waypt_pub')

		#Set up QoS Profiles for passing images over WiFi
		image_qos_profile = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			durability=QoSDurabilityPolicy.VOLATILE,
			depth=1
		)

		self.feedback_sub = self.create_subscription(NavigateToPose_FeedbackMessage,'/navigate_to_pose/_action/feedback',self.feedback_callback,10)
		self.feedback_sub
		
		#Declare waypt_publisher
		self.waypt_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
		
	def feedback_callback(self,msg):
		feedback = msg.status.status
		self.get_logger().info('feedback: "%s"'% feedback)
		STATUS_SUCCEEDED = "Waypoint Reached" #change status_succeeded to whatever output says the waypoint is reached

		goal = PoseStamped()
		if state == 1:
			if feedback == STATUS_SUCCEEDED:
				state = 2
			else:
				self.get_logger().info('state: "%s"'% state)
				goal.header.stamp = self.get_clock().now().to_msg()    #rclpy.Time.now()
				goal.header.frame_id = "map"
				goal.pose.position.x = 3.9
				goal.pose.position.y = -0.85
				goal.pose.position.z = 0.0

				goal.pose.orientation.x = 0.0
				goal.pose.orientation.y = 0.0
				goal.pose.orientation.z = 0.0
				goal.pose.orientation.w = 1.0
				self.waypt_pub.publish(goal)
		elif state == 2:
			if feedback == STATUS_SUCCEEDED:
				state = 3
			else:
				self.get_logger().info('state: "%s"'% state)
				goal.header.stamp = self.get_clock().now().to_msg()    #rclpy.Time.now()
				goal.header.frame_id = "map"
				goal.pose.position.x = 0.0
				goal.pose.position.y = 0.0
				goal.pose.position.z = 0.0

				goal.pose.orientation.x = 0.0
				goal.pose.orientation.y = 0.0
				goal.pose.orientation.z = 0.0
				goal.pose.orientation.w = 1.0
				self.waypt_pub.publish(goal)
		elif state == 3:
			if feedback == STATUS_SUCCEEDED:
				state = 4
			else:
				self.get_logger().info('state: "%s"'% state)
				goal.header.stamp = self.get_clock().now().to_msg()    #rclpy.Time.now()
				goal.header.frame_id = "map"
				goal.pose.position.x = 3.9
				goal.pose.position.y = -0.85
				goal.pose.position.z = 0.0

				goal.pose.orientation.x = 0.0
				goal.pose.orientation.y = 0.0
				goal.pose.orientation.z = 0.0
				goal.pose.orientation.w = 1.0
				self.waypt_pub.publish(goal)
		else:
				self.get_logger().info('All waypoints reached')
	

def main(args=None):
	# Setting up publisher values
	rclpy.init(args=args)
	feedback_sub=WayptPub()
	rclpy.spin(feedback_sub)
	rclpy.shutdown()
