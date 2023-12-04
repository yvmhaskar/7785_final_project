# Joseph Sommer and Yash Mhaskar

# aligning
# get_lidar data and function. Already have from lab 4
# state 3, after detecting a wall, what do you do?

# Required Libraries
from __future__ import print_function
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import math
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

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
from cv_bridge import CvBridge

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
w_des = 0.0 ## Beginning orientation
global prediction
prediction = 0
global lidar_input
lidar_input = 0
global predicted
predicted = np.zeros([20,1])

global block_centers
#block_centers = np.array([[-0.25,0.0],[0.6,0.0],[0.6,-0.9],[-0.25,-0.9],[-0.25,-1.8],[-1.1,-1.8],[-1.1,-0.9],[-1.1,0.0],
#						  [-2.0,0.0],[-2.0,-0.9],[-2.0,-1.8],[-2.9,-1.8],[-2.9,-0.9],[-2.9,0.0],[-3.8,0.0],[-3.8,-0.9],[-3.8,-1.8]])

block_centers = np.array([[0.0,0.0], [0.9, 0], [0.9,-0.9], [0.0,-0.9], [0.0, -1.8], [-0.9, -1.8], [-0.9, -0.9], [-0.9, 0.0],[-1.8, 0.0], 
						  [-1.8, -0.9], [-1.8, -1.8], [-2.7, -1.8],[-2.7, -0.9], [-2.7, 0.0], [-3.6, 0.0], [-3.6, -0.9],[-3.6, 0.0]])
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

		#Declare image_callback node is subcribing to the /camera/image/compressed topic.
		self.get_image = self.create_subscription(CompressedImage,'/image_raw/compressed',self.image_callback,image_qos_profile)
		self.get_image # Prevents unused variable warning.

		#Declare feedback_sub is subscribing to the /navigate_to_pose/_action/feedback.
		self.feedback_sub = self.create_subscription(NavigateToPose_FeedbackMessage,'/navigate_to_pose/_action/feedback',self.feedback_callback,10)
		self.feedback_sub
		
		#Declare lidar_sub is subscribing to the wall_detection.
		self.lidar_sub = self.create_subscription(Float32,'/wall_detection',self.lidar_callback,10)
		self.lidar_sub

		#Declare waypt_publisher
		self.waypt_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)

	def image_callback(self, CompressedImage):
		global flag_get_image, image_array, state, predicted, prediction

		if flag_get_image>0 and flag_get_image<=19:
			
			image = [CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")]
			predicted[flag_get_image] = self.classify_image(image)
			flag_get_image = flag_get_image+1
			#if flag_get_image==20:
			#	prediction = np.median(predicted)
				#cv2.imshow('CHAIN_APPROX_SIMPLE Point only', image)
			#	self.get_logger().info('Prediction: {}'.format(prediction))
		else:
			flag_get_image=0

	def lidar_callback(self,msg):
		global lidar_input
		msg_data = msg.data
		lidar_input = float(msg_data)

	def feedback_callback(self,msg):
		global state, x_des, y_des, w_des, flag_get_image, image_array, block_centers, prediction, lidar_input, predicted
		
		feedback = msg.feedback
		self.x_cur = feedback.current_pose.pose.position.x
		self.y_cur = feedback.current_pose.pose.position.y
		self.w_cur = feedback.current_pose.pose.orientation.w


		#List of block centers

		bc_shape = block_centers.shape[0]
		# structured like poseStamped so calculate own error and use to determine states

		err_lim_lin = 0.1 # error limit linear
		err_lim_ang = 0.5 # error limit angular

		if state == 0:
			
			#dist_err = np.zeros([bc_shape,1])
			#i = 0
			min_error = float('inf')
			newWaypoint = np.zeros(3)
			for waypoint in block_centers:
				x_target, y_target = waypoint #took out w_temp

				# Calculate error for each waypoint
				error = math.sqrt((x_target - self.x_cur) ** 2 + (y_target - self.y_cur) ** 2)
				if error < min_error:
					min_error = error
				newWaypoint[:2] = waypoint
			#for bc in block_centers:
			#	dist_err[i] = math.sqrt((bc[0]-self.x_cur)**2 + (bc[1]-self.y_cur)**2)
			#	i=i+1
			#idx = np.argmin(dist_err)

			x_des = newWaypoint[0]
			y_des = newWaypoint[1]
			
			#y_des = block_centers[idx][1]
			#err_lin = np.min(dist_err)
			err_lin = min_error
			if err_lin<err_lim_lin: # pick a new value based off burger.yaml file/tuning
				state = 2
				self.get_logger().info('State 2: Taking images')
				flag_get_image = 1
			
		if state == 1:
			self.get_logger().info('State 1: Aligning with the image')
			
			##### ALIGNING CODE#### image_array

			x_des = self.x_cur
			y_des = self.y_cur
			#w_des = GET FROM ALIGNING CODE
			#err_ang = err in angle
			if err_ang<err_lim_ang: # pick a new value based off burger.yaml file/tuning
				state = 2
				flag_get_image = 1

		if state == 2:
			
			if flag_get_image==0:
				self.get_logger().info('State 2: Classified image')
				#prediction = self.classify_image(image_array)
				prediction = np.median(predicted)
				self.get_logger().info('Prediction: {}'.format(prediction))
				self.get_logger().info('State 2: Classifying image')
				
				if prediction==0 and lidar_input==1:# 1 means wall in front
					self.get_logger().info('State 2: classified wall')
					state = 3
					self.get_logger().info('State 3: Wall detected, turning right')

					w_des = self.w_cur + 3.14/2
				else:
					state = 4
					self.get_logger().info('State 4: Determining Next Waypoint')
		
		if state == 3:
			
			
			err_ang = abs(w_des-self.w_cur)
			if err_ang<err_lim_ang: # pick a new value based off burger.yaml file/tuning
				state = 2# change depending on status of alignment
				flag_get_image = 1
			
		if state == 4:
			
			if prediction==0 and lidar_input==0: # wall classification without wall there
				Newpt = self.newWaypt(self.x_cur,self.y_cur,self.w_cur,0)
				x_des = Newpt[0]
				y_des = Newpt[1]
				w_des = Newpt[2]
				state = 5
				self.get_logger().info('State 5: Moving to straight waypoint')

			if prediction==1 or prediction==2: # left arrows
				Newpt = self.newWaypt(self.x_cur,self.y_cur,self.w_cur,2)
				x_des = Newpt[0]
				y_des = Newpt[1]
				w_des = Newpt[2]
				state = 5
				self.get_logger().info('State 5: Moving to left waypoint')

			if prediction==3 or prediction==4: # right arrows
				Newpt = self.newWaypt(self.x_cur,self.y_cur,self.w_cur,1)
				x_des = Newpt[0]
				y_des = Newpt[1]
				w_des = Newpt[2]
				state = 5
				self.get_logger().info('State 5: Moving to right waypoint')

			if prediction==5 or prediction==6: # stop an turn around
				Newpt = self.newWaypt(self.x_cur,self.y_cur,self.w_cur,3)
				x_des = Newpt[0]
				y_des = Newpt[1]
				w_des = Newpt[2]
				state = 5
				self.get_logger().info('State 5: Moving to back waypoint')

			if prediction==7: # reached goal
				state = 6
				x_des = self.x_cur
				y_des = self.y_cur
				w_des = self.w_cur

		if state == 5:
			
			dist_err = math.sqrt((x_des-self.x_cur)**2 + (y_des-self.y_cur)**2)

			if dist_err< err_lim_lin:
				state=2
				self.get_logger().info('State 2: Classifying')
				
		if state == 6:
			self.get_logger().info('State 6: REACHED GOALLLLLLL')

		#goal = PoseStamped()
		#goal.header.frame_id = "map"
		#goal.pose.position.x = x_des
		#goal.pose.position.y = y_des
		#goal.pose.position.z = 0.0
		#goal.pose.orientation.x = 0.0
		#goal.pose.orientation.y = 0.0
		#goal.pose.orientation.z = 0.0
		#goal.pose.orientation.w = w_des
		#self.waypt_pub.publish(goal)

		#self.waypt_pub.publish(goal)
		
	def classify_image(self,image):
		
		# Load data
		data = image

		# Classifier Filepath
		classifier_filepath = '/home/ymhaskar/classifier.pkl'
		
		# Crop images
		data_cropped = [self.crop_function(image) for image in data]

		# Feature extraction using color histogram
		color_hist_features = [self.extract_color_histogram(image).reshape(-1) for image in data_cropped]
		
		# Feature extraction using HOG
		fixed_size = (308,410)
		resized_images = [cv2.resize(image, fixed_size) for image in data_cropped]
		hog_features = [self.extract_hog_features(image).reshape(-1) for image in resized_images]
		
		# Feature extraction using HOG skewness
		skewness_features = [self.compute_hog_skewness(image) for image in data]
		
		# Concatenate features
		features = np.hstack((color_hist_features, hog_features, skewness_features))
		
		# Load Classifier
		classifier = joblib.load(classifier_filepath)
		predicted = classifier.predict(features)
		return predicted
	
	# Cropping the Images based on the colours
	def crop_function(self,image):
		low_H = 0
		low_S = 99
		low_V = 119
		high_H = 10
		high_S = 255
		high_V = 236
		min_size = 0.1
		no_contour = 0
		image = image[0:410, 50:258]
		blur = cv2.GaussianBlur(image,(15,15),0)
		blur_HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		frame_threshold = cv2.inRange(blur_HSV,(low_H, low_S, low_V),(high_H, high_S, high_V))
		frame_threshold = self.apply_closing(frame_threshold, 35)
		contours = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if len(contours[0])>0:
			#print(len(contours))
			filtered_contours = [contour for contour in contours[0] if cv2.contourArea(contour) > min_size]
			if len(filtered_contours)==0:
				no_contour = 1
		else:
			no_contour = 1
		if no_contour==1:
			low_H = 20
			low_S = 90
			low_V = 0
			high_H = 126
			high_S = 255
			high_V = 255

			frame_threshold = cv2.inRange(blur_HSV,(low_H, low_S, low_V),(high_H, high_S, high_V))
			frame_threshold = self.apply_closing(frame_threshold, 35)
			contours = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			if len(contours[0])>0:
				filtered_contours = [contour for contour in contours[0] if cv2.contourArea(contour) > min_size]
				if len(filtered_contours)==0:
					no_contour = 2
			else:
				no_contour = 2

		if no_contour != 2:
			count = filtered_contours[0]
			(x_axis,y_axis),radius = cv2.minEnclosingCircle(count)
			center = (int(x_axis),int(y_axis))
			radius = int(radius)
			# reduces likelihood of showing contour on wrong object
			if radius>10 and radius*1.2<center[0] and radius*1.2<center[1]:# from 10
					#cv2.circle(image,center,radius,(0,255,0),2)
					#cv2.circle(frame_threshold,center,radius,(0,255,0),2)
				x1,y1,x2,y2 = int(center[0]-radius*1.2), int(center[1]-radius*1.2), int(center[0]+radius*1.2), int(center[1]+radius*1.2)
				if abs(x2-x1)>=20 and abs(y1-y2)>=20:
					cropped = image[int(y1):int(y2), int(x1):int(x2)]
				else:
					#print("No Contour found")
					cropped = image
			else:
				#print("No Contour found")
				cropped = image
		else:
			#print("No Contour found")
			cropped = image
		#cv2.imshow('CHAIN_APPROX_SIMPLE Point only', blur_HSV)
		print(no_contour)
		#cv2.waitKey(0)
		#print(no_contour)
		return cropped
	
	# function that takes current point and uses desired direction to output a new position and orientation
	def newWaypt(self,x, y, w, dir):
		global block_centers
		waypoints = block_centers
		block_size = 0.9

		if dir == 1: #Right
			dw = -np.pi / 2.0
			local_p1 = np.array([0, block_size, 0, 1])
		elif dir == 2: #Left
			dw = np.pi / 2.0
			local_p1 = np.array([0, -1*block_size, 0, 1])
		elif dir == 3: #Back
			dw = np.pi
			local_p1 = np.array([-1*block_size, 0, 0, 1])
		elif dir == 0: #Forward
			dw = 0.0
			local_p1 = np.array([block_size, 0, 0, 1])
		# Calculate Pnew once outside the loop
		T = np.array([
			[math.cos(w), -math.sin(w), 0, x],
			[math.sin(w), math.cos(w), 0, y],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
		])
		Pnew = np.dot(T, local_p1)
		min_error = float('inf')
		newWaypoint = np.zeros(3)

		for waypoint in waypoints:
			x_target, y_target = waypoint #took out w_temp

			# Calculate error for each waypoint
			error = math.sqrt((x_target - Pnew[0]) ** 2 + (y_target - Pnew[1]) ** 2)
			if error < min_error:
				min_error = error
				newWaypoint[:2] = waypoint

		

		Wnew = w+dw
		Wnew = np.arctan2(np.sin(Wnew), np.cos(Wnew))
		newWaypoint[2] = Wnew
		#Pnew[2] = Wnew
		self.get_logger().info('newWaypoint: {}'.format(w))
		return newWaypoint
	
	def apply_closing(self,image, kernel_size):
		kernel = np.ones((kernel_size,kernel_size), np.uint8)
		closing_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
		return closing_result

	# Function to extract color histogram features
	def extract_color_histogram(self, image):
		blur = cv2.GaussianBlur(image,(15,15),0)
		image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  
		#print('AVG HSV')
		#h, s, v = cv2.split(image)
		avg_h = np.mean(image[:,:,0])/179.0
		avg_s = np.mean(image[:,:,1])/255.0
		avg_v = np.mean(image[:,:,2])/255.0
		avg_h = np.nan_to_num(avg_h)
		avg_s = np.nan_to_num(avg_s)
		avg_v = np.nan_to_num(avg_v)
		#print(avg_h)
		return np.array([avg_h, avg_s, avg_v])

	# Function to extract SIFT features
	def extract_sift_features(self,image):
		gray_image = color.rgb2gray(image)
		sift = SIFT()
		sift.detect_and_extract(gray_image)
		return sift.descriptors.flatten()

	# Function to extract HOG features
	def extract_hog_features(self,image):
		gray_image = color.rgb2gray(image)
		fd, hog_image = hog(
			gray_image,
			orientations=8,
			pixels_per_cell=(32, 32),
			cells_per_block=(1, 1),
			visualize=True,
			block_norm='L2-Hys'#,
			#multichannel=False
		)
		hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
		hog_image_np = hog_image_rescaled.flatten()
		hog_image_np = hog_image_np.astype(np.float32)
		return hog_image_np*10000.0
		# Function to extract skewness of HOG

	def compute_hog_skewness(self,image):
		gray_image = color.rgb2gray(image)
		fd, hog_image = hog(
			gray_image,
			orientations=8,
			pixels_per_cell=(32, 32),
			cells_per_block=(1, 1),
			visualize=True,
			block_norm='L2-Hys'#,
			#multichannel=False
		)
		hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
		hog_image_np = hog_image_rescaled.flatten()
		skewness = skew(hog_image_np)
		return np.array([skewness])

	# normalize an array
	def normalize_Skew(self, skew_vals):
		norm_vals = np.zeros(len(skew_vals))
		for i in range(0,len(skew_vals)):
			norm_vals[i] = (skew_vals[i]-min(skew_vals))/(max(skew_vals)-min(skew_vals))
		return norm_vals
	
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
		#self.get_logger().info('X: {}'.format(x))
		#self.get_logger().info('Y: {}'.format(y))
		#self.get_logger().info('W: {}'.format(w))

def main(args=None):
	
	#rclpy.init(args=args)
	#nav_maze=NavMaze()
	#nav_maze.feedback_callback
	#rclpy.spin(nav_maze)
	#nav_maze.destroy_node()
	#rclpy.shutdown()

	global x_des, y_des, w_des
	rclpy.init(args=args)
	nav_maze=NavMaze()
	print('Started Publishing')
	while rclpy.ok():
		nav_maze.feedback_callback
		nav_maze.publish_waypt(x_des, y_des, w_des)
		rclpy.spin_once(nav_maze, timeout_sec=1)

	rclpy.shutdown()

	rclpy.shutdown()
