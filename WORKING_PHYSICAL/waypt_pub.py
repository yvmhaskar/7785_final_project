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
global x_des
x_des = 0.0
global y_des
y_des = 0.0
global w_des
w_des = 1.0

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
		global state, x_des, y_des, w_des
		feedback = msg.feedback
		#self.get_logger().info('feedback: "%s"'% feedback)
		#self.distance_remaining = feedback.distance_remaining
		self.x_cur = feedback.current_pose.pose.position.x
		self.y_cur = feedback.current_pose.pose.position.y
		self.w_cur = feedback.current_pose.pose.orientation.w

		#self.get_logger().info('Remaining distance: "%s"'% self.distance_remaining)
		 # structured like poseStamped so calculate own error and use to determine states

		err_ran = 0.1 # error range

		if state == 1:
			x_des = 0.3#3.9
			y_des = 1.5#-0.85
			err =  math.sqrt((x_des-self.x_cur)**2 + (y_des-self.y_cur)**2)
			self.get_logger().info('error: "%s"'% err)
			if err<err_ran: # pick a new value based off burger.yaml file/tuning
				state = 2
		elif state == 2:
			x_des = 1.2#3.31
			y_des = 0.14#0.79
			err = math.sqrt((x_des-self.x_cur)**2 + (y_des-self.y_cur)**2)
			self.get_logger().info('error: "%s"'% err)
			if err<err_ran: # pick a new value based off burger.yaml file/tuning
				state = 3
		elif state == 3:
			x_des = 2.0#0.0
			y_des = 0.75#0.0
			err = math.sqrt((x_des-self.x_cur)**2 + (y_des-self.y_cur)**2)
			self.get_logger().info('error: "%s"'% err)
			if err<err_ran: # pick a new value based off burger.yaml file/tuning
				state = 4
		else:
				self.get_logger().info('All waypoints reached')
		#self.waypt_pub.publish(goal)
		
	def publish_waypt(self,x,y,w):
		goal = PoseStamped()
		goal.header.frame_id = "map"
		goal.pose.position.x = x
		goal.pose.position.y = y
		goal.pose.position.z = 0.0
		goal.pose.orientation.x = 0.0
		goal.pose.orientation.y = 0.0
		goal.pose.orientation.z = 0.0
		goal.pose.orientation.w = w
		self.waypt_pub.publish(goal)
		print('published')

def main(args=None):

	# Setting up publisher values
	global x_des, y_des, w_des
	rclpy.init(args=args)
	waypt_pub=WayptPub()
	print('Started Publishing')
	while rclpy.ok():
		waypt_pub.feedback_callback
		waypt_pub.publish_waypt(x_des, y_des, w_des)
		rclpy.spin_once(waypt_pub, timeout_sec=1)

	rclpy.shutdown()