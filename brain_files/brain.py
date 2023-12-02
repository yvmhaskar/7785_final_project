# Joseph Sommer and Yash Mhaskar

# Required Libraries
from __future__ import print_function
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import math

# numpy
import numpy as np

# scikit-learn (some common classifiers are included as examples, feel free to add your own)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals import joblib
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# cv2
import cv2

# scikit-image
from skimage import transform, color, exposure
from skimage.feature import SIFT, hog

# scipy
from scipy.stats import skew

# Global Variables
global state
state = 0
global flag_get_image
flag_get_image = 0
global x_des
x_des = 0.0
global y_des
y_des = 0.0
global w_des
w_des = 1.0
global image_array

class NavMaze(Node):
	def __init__(self):
		super().__init__('nav_maze')

		#Set up QoS Profiles for passing images over WiFi
		image_qos_profile = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			durability=QoSDurabilityPolicy.VOLATILE,
			depth=1
		)

		#Declare get_image node is subcribing to the /camera/image/compressed topic.
		self.get_image = self.create_subscription(CompressedImage,'/image_raw/compressed',self.image_callback,image_qos_profile)
		self.get_image # Prevents unused variable warning.

        #Declare feedback_sub is subscribing to the /navigate_to_pose/_action/feedback.
		self.feedback_sub = self.create_subscription(NavigateToPose_FeedbackMessage,'/navigate_to_pose/_action/feedback',self.feedback_callback,10)
		self.feedback_sub
		
		#Declare waypt_publisher
		self.waypt_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)

    def get_image(self, CompressedImage):
        global flag_get_image
        global image_array
        if flag_get_image>0 and flag_get_image<=4:
            image_array[flag_get_image-1] = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
            flag_get_image = flag_get_image+1
        else:
            flag_get_image=0

	def feedback_callback(self,msg):
		global state, x_des, y_des, w_des, flag_get_image

		feedback = msg.feedback
		self.x_cur = feedback.current_pose.pose.position.x
		self.y_cur = feedback.current_pose.pose.position.y
		self.w_cur = feedback.current_pose.pose.orientation.w


        #List of block centers
        block_centers = np.array[[1.0,1.0],[2.0,2.0],[3.0,3.0]]
        bc_shape = block_centers.shape
        # structured like poseStamped so calculate own error and use to determine states

        err_ran = 0.1 # error range

        if state == 0:
        	self.get_logger().info('State 0: Moving to center of block')
            dist_err = np.zeros(bc_shape)
            i = 0
            for bc in block_centers:
                dist_err[i] = math.sqrt((bc-self.x_cur)**2 + (bc-self.y_cur)**2)
                i=i+1
            idx = np.argmin(dist_err)
            x_des = block_centers[idx][0]
            y_des = block_centers[idx][1]
            self.get_logger().info('error: "%s"'% err)
			if err<err_ran: # pick a new value based off burger.yaml file/tuning
				state = 1
                flag_get_image = 1

		if state == 1:
            self.get_logger().info('State 1: Classifying image')
            if image_classified:
                state = 3
            else:
                state = 2
		
        if state == 2:
			self.get_logger().info('State 2: No image, turning right')
            
        if state == 3:

            x_des = 1.2   #3.31   #1.2#3.31
			y_des = 0.14 #0.79 #0.14#0.79
			err = math.sqrt((x_des-self.x_cur)**2 + (y_des-self.y_cur)**2)
			self.get_logger().info('error: "%s"'% err)
			if err<err_ran: # pick a new value based off burger.yaml file/tuning
				state = 3
		elif state == 3:
			x_des = 2.0 #2.0#0.0
			y_des = 0.75 #0.75#0.0
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
	nav_maze=NavMaze()
	print('Started Publishing')
	while rclpy.ok():
		nav_maze.feedback_callback
		nav_maze.publish_waypt(x_des, y_des, w_des)
		rclpy.spin_once(waypt_pub, timeout_sec=1)

	rclpy.shutdown()