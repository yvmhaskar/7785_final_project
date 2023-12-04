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
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

global predicted
predicted = np.zeros([20,1])
global flag_get_image
flag_get_image = 0
global got_predict
got_predict=0
global align_x
align_x= 0
global flag_align_robot
flag_align_robot = 0
global alpha	
alpha = 0
global counter
counter=0
global odom_err
odom_err=0
#max velocity
MAX_VEL = 0.15 #m/s
MAX_ANG = 1.5 #rad/s

#PID constants
KW = 0.8

#LIDAR forward scan segment
LIDAR_FOV = np.pi/8 #rad

#tolerance
D_TOL1 = 0.8
D_TOL2 = 0.6 #m
A_TOL = 0.02 #rad

WIDTH, HEIGHT = 256, 144 #144p
MIN_WHITE = 240

def normalize(angle):
	return np.arctan2(np.sin(angle), np.cos(angle))

class SolveMaze(Node):
	def __init__(self):
		super().__init__('SolveMaze') #create node

		self.reset_frame = True #reset global frame to current position/orientation

		#subcribe to /odom
		self._odom_subscriber = self.create_subscription(
			Odometry,
			'/odom',
			self._odom_callback,
			1)
		self._odom_subscriber # Prevents unused variable warning.

		#Set up QoS Profiles for passing images over WiFi
		image_qos_profile = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			durability=QoSDurabilityPolicy.VOLATILE,
			depth=1
		)
		
		# Create a subscriber which receive scan messages 
		# from LIDAR in the '/scan'
		# receive every data scanned
		self._LIDAR_subscriber = self.create_subscription(
			LaserScan,
			'/scan',
			self._LIDAR_callback,
			image_qos_profile
		)

		#subcribe to /image_raw/compressed
		self._video_subscriber = self.create_subscription(
				CompressedImage,
				'/image_raw/compressed',
				self._image_callback,
				image_qos_profile)
		self._video_subscriber # Prevents unused variable warning.

		#publish twist commands
		self._vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1)

	
	#--- dispatcher ---
	def solve_maze(self):
		global predicted, flag_get_image, got_predict, flag_align_robot, counter, odom_err
		while True:
			self.wait_image()
			if got_predict == 1:
				pred = np.median(predicted) # change to class
				self.get_logger().info(f'prediction: {pred}')
				ang_turn = 0.0
				if (pred == 3.0 or pred == 4.0) and alpha>0:
					ang_turn = -1* abs(alpha)
				elif (pred == 3.0 or pred == 4.0) and alpha<0:
					ang_turn = abs(alpha)
				elif (pred == 1.0 or pred == 2.0) and alpha>0:
					ang_turn = abs(alpha)
				elif (pred == 1.0 or pred == 2.0) and alpha<0:
					ang_turn = -1* abs(alpha)
				elif (pred == 5.0 or pred == 6.0) and alpha>0:
					ang_turn = -1*abs(alpha)
				elif (pred == 5.0 or pred == 6.0) and alpha<0:
					ang_turn = abs(alpha)

				
				if pred % 1 != 0:
					pred = 0

				if counter == 5:
					pred = 3.0

				if pred == 0:
					self.move_forward(D_TOL1)
					flag_align_robot = 1
					while flag_align_robot:
						rclpy.spin_once(self)
					self.pub_coord(align_x)
					self.move_forward(D_TOL2)
					counter = counter+1

				elif pred == 1.0 or pred == 2.0:
					odom_err=odom_err+0.002
					self.rotate(ang_turn)#np.pi/2-odom_err)
					self.move_forward(D_TOL1)
					flag_align_robot = 1
					while flag_align_robot:
						rclpy.spin_once(self)
					self.pub_coord(align_x)
					self.move_forward(D_TOL2)
					counter=0

				elif pred == 3.0 or pred == 4.0:
					odom_err=odom_err+0.002
					self.rotate(ang_turn)#-np.pi/2+odom_err)
					self.move_forward(D_TOL1)
					flag_align_robot = 1
					while flag_align_robot:
						rclpy.spin_once(self)
					self.pub_coord(align_x)
					self.move_forward(D_TOL2)
					counter=0

				elif pred == 5.0 or pred == 6.0:
					odom_err=odom_err+0.002
					self.rotate(ang_turn)#np.pi-2*odom_err)
					self.move_forward(D_TOL1)
					flag_align_robot = 1
					while flag_align_robot:
						rclpy.spin_once(self)
					self.pub_coord(align_x)
					self.move_forward(D_TOL2)
					counter=0

				elif pred == 7.0:
					print('REACHED GOAL')
				else:
					return
				flag_get_image = 0
				got_predict=0

	#--- atomic move functions ---
	def move_forward(self, d_range):
		print('forward')
		global align_x
		while True:
			self.wait_odom()

			if self.new_lidar:
				self.new_lidar = False
				if self.obst_dist <= d_range:
					self._vel_publisher.publish(Twist())
					return

			cmd = Twist()
			cmd.linear.x = MAX_VEL
			cmd.angular.z = np.clip(-KW*self.a, -MAX_ANG, MAX_ANG)
			self._vel_publisher.publish(cmd)

	def rotate(self, angle):
		print("reached rotate")
		print(angle)
		while True:
			self.wait_odom()

			err_a = normalize(angle - self.a)
			if abs(err_a) <= A_TOL:
				self._vel_publisher.publish(Twist())
				self.reset_frame = True
				return

			cmd = Twist()
			cmd.angular.z = np.clip(KW*err_a, -MAX_ANG, MAX_ANG)
			self._vel_publisher.publish(cmd)

	def pub_coord(self, x1):
		print('Aligning')
		global alpha		
		#logic
		# Set direction. Positive is angled right, negative is angled left
		direction = 1
		if x1 > 328/2:
			direction = -1
		# Perpendicular distance in pixels to center and outer edges of object
		x1_bar = 328/2 - x1 # units of pixel values. Describes perpendicular distance from x_axis of robot
		x1_bar = x1_bar*direction

		#x2_bar = x1_bar - r # result is perp distance from center axis to edge of object. In pixels
		#x3_bar = x1_bar + r # result is perp distance from center axis to other edge of object. In pixels

		# angular offset from x axis to center and outer edges of object
		theta1 = x1_bar * (62.2/2) / (328/2) # result is angular displacement from x_axis of robot in degrees
		#theta2 = x2_bar * (62.2/2) / (328/2) # result is angular displacement from x_axis of robot in degrees
		#theta3 = x3_bar * (62.2/2) / (328/2) # result is angular displacement from x_axis of robot in degrees

		#ang_err = direction*(angle_index2+index) * angular_resolution # angular error in degrees
		ang_err = theta1*direction* np.pi/180.0
		#ang_err = (ang_err + ang_err_old)/2
		# publish direction
		#ang_err_old = ang_err
		# Publish the x-axis position
		if abs(ang_err)<0.5:
			self.rotate(ang_err)
		else:
			return
		return ang_err

	#--- wait for topic updates ---
	def wait_odom(self):
		self.new_odom = False
		while not self.new_odom:
			rclpy.spin_once(self)

	def wait_image(self):
		global flag_get_image
		self.new_img = False
		flag_get_image=1
		while not self.new_img:
			rclpy.spin_once(self)

	def _image_callback(self, CompressedImage):
			# The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"		
		global flag_get_image, predicted, got_predict, flag_align_robot
		
		if flag_align_robot==1:
			image = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
			p = self.classify_image(image)

		if flag_get_image>0 and flag_get_image<=20:
				image = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
				predicted[flag_get_image-1] = self.classify_image(image)
				if flag_get_image == 20:
					got_predict=1
				flag_get_image = flag_get_image+1
		else:
			flag_get_image=0
			self.new_img = True

	##### Image Processing & classification
	def classify_image(self,image):
		
		# Load data
		data = [image]
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
		global flag_align_robot, align_x
		low_H = 0
		low_S = 99
		low_V = 119
		high_H = 10
		high_S = 255
		high_V = 236
		min_size = 0.1
		no_contour = 0
		#image = image[0:410, 50:258]
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
			if flag_align_robot==1:
				align_x = x_axis
				flag_align_robot=0
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
		#print(no_contour)
		#cv2.waitKey(0)
		#print(no_contour)
		return cropped
	
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
	
	#--- topic callbacks ---
	def _odom_callback(self, odom_msg):
		position = odom_msg.pose.pose.position
		q = odom_msg.pose.pose.orientation
		yaw = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

		if self.reset_frame:
			Mrot = np.matrix([[np.cos(yaw), np.sin(yaw)],[-np.sin(yaw), np.cos(yaw)]])
			self.init_a = yaw
			self.init_x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
			self.init_y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
			self.reset_frame = False

		Mrot = np.matrix([[np.cos(self.init_a), np.sin(self.init_a)],[-np.sin(self.init_a), np.cos(self.init_a)]])
		self.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.init_x
		self.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.init_y
		self.a = normalize(yaw - self.init_a)
		self.new_odom = True

	def _LIDAR_callback(self, msg):
		scan_dists = np.array(msg.ranges)
		scan_angles = np.linspace(msg.angle_min, msg.angle_max, len(scan_dists))
		scan_angles[scan_angles > np.pi] -= 2*np.pi
		scan_dists = scan_dists[(scan_angles > -LIDAR_FOV/2) & (scan_angles < LIDAR_FOV/2)]

		i = np.nanargmin(scan_dists)
		self.obst_dist = scan_dists[i]
		self.new_lidar = True

	

def main():
	rclpy.init() #init routine needed for ROS2.
	node = SolveMaze() #Create class object to be used.
	node.solve_maze()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
