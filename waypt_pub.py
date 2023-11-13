# Joseph Sommer and Yash Mhaskar

from __future__ import print_function
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import math

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
		global state
		global reached
		feedback = msg.feedback
		#self.get_logger().info('feedback: "%s"'% feedback)
		distance_remaining = feedback.distance_remaining
		x_cur = feedback.current_pose.pose.position.x
		y_cur = feedback.current_pose.pose.position.y
		w = feedback.current_pose.pose.orientation.w

		self.get_logger().info('Remaining distance: "%s"'% distance_remaining)
		 # structured like poseStamped so calculate own error and use to determine states
		
		err_ran = 0.02 # error range

		goal = PoseStamped()
		goal.header.stamp = self.get_clock().now().to_msg()    #rclpy.Time.now()
		goal.header.frame_id = "map"

		goal.pose.position.z = 0.0
		goal.pose.orientation.x = 0.0
		goal.pose.orientation.y = 0.0
		goal.pose.orientation.z = 0.0
		goal.pose.orientation.w = 1.0
		
		if state == 1:
			x_des = 3.9
			y_des = -0.85
			err =  math.sqrt((x_des-x_cur)**2 + (y_des-y_cur)**2)
			self.get_logger().info('error: "%s"'% err)
			if err<err_ran: # pick a new value based off burger.yaml file/tuning
				state = 2
			else:
				self.get_logger().info('state: "%s"'% state)
				goal.pose.position.x = x_des
				goal.pose.position.y = y_des
				
		elif state == 2:
			x_des = 0.0
			y_des = 0.0
			err = math.sqrt((x_des-x_cur)**2 + (y_des-y_cur)**2)
			self.get_logger().info('error: "%s"'% err)
			if err<err_ran: # pick a new value based off burger.yaml file/tuning
				state = 3
			else:
				self.get_logger().info('state: "%s"'% state)
				goal.pose.position.x = x_des
				goal.pose.position.y = y_des

		elif state == 3:
			x_des = 3.9
			y_des = -0.85
			err = math.sqrt((x_des-x_cur)**2 + (y_des-y_cur)**2)
			self.get_logger().info('error: "%s"'% err)
			if err<err_ran: # pick a new value based off burger.yaml file/tuning
				state = 4
			else:
				self.get_logger().info('state: "%s"'% state)
				goal.header.stamp = self.get_clock().now().to_msg()    #rclpy.Time.now()
				goal.header.frame_id = "map"
				goal.pose.position.x = x_des
				goal.pose.position.y = y_des
		else:
				self.get_logger().info('All waypoints reached')
		self.waypt_pub.publish(goal)
	

def main(args=None):
	# Setting up publisher values
	rclpy.init(args=args)
	waypt_pub=WayptPub()
	print('Running')
	

	goal = PoseStamped()
	goal.header.frame_id = "map"
	goal.pose.position.x = 3.9
	goal.pose.position.y = -0.085
	goal.pose.position.z = 0.0
	goal.pose.orientation.x = 0.0
	goal.pose.orientation.y = 0.0
	goal.pose.orientation.z = 0.0
	goal.pose.orientation.w = 1.0

	for x in range(100):
		waypt_pub.waypt_pub.publish(goal)
		print('published')

	rclpy.spin(waypt_pub)
	rclpy.shutdown()