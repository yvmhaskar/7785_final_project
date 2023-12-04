import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
import cv2

import torch
from torchvision import transforms
import torch.nn as nn
from torch import flatten

#max velocity
MAX_VEL = 0.2 #m/s
MAX_ANG = 1.5 #rad/s

#PID constants
KW = 0.8

#LIDAR forward scan segment
LIDAR_FOV = np.pi/6 #rad

#tolerance
D_TOL = 0.45 #m
A_TOL = 0.02 #rad

WIDTH, HEIGHT = 256, 144 #144p
MIN_WHITE = 240

PTH = 'model_weights.pth' #path to saved CNN weights
transform = transforms.Compose([transforms.Resize(size=(144, 256)), #144p
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#from pytorch docs
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, #RGB image input
                               out_channels=20,
                               kernel_size=(10, 10))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(6, 6), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=20, #output of conv1
                               out_channels=50,
                               kernel_size=(10, 10))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(10, 10), stride=(5, 5))

        self.fc1 = nn.Linear(in_features=3900, #how do you get this
                             out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, #output of fc1
                             out_features=num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.logSoftmax(self.fc2(x))
        return x

def normalize(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def preprocess(image):
    img = cv2.resize(image, (WIDTH, HEIGHT)) #resize to spec

    #--- whitebalance with grayworld assumption ---
    grayworld = np.float32(255*(img/img.mean(axis=(0,1))).clip(0,1))

    #--- find initial roi, white out excess ---
    blur = cv2.GaussianBlur(grayworld, (69,69), 0) #blur image to reduce noise
    grayscale = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(grayscale, MIN_WHITE, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(white_mask))
    y, h = HEIGHT - (y+h), 2*(y+h) - HEIGHT #set mask top edge to mirror bottom edge
    roi_img = 255*np.ones_like(img) #white background
    roi_img[y:y+h, x:x+w] = grayworld[y:y+h, x:x+w]

    #--- ? ---
    blur = cv2.GaussianBlur(roi_img, (5,5), 0) #blur image to reduce noise
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) #convert to HSV for improved color segmentation
    _, saturation, _ = cv2.split(hsv_img)
    _, sat_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)

    #--- ? ---
    return cv2.bitwise_and(img, img, mask=sat_mask)

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

        self.model = LeNet(6)
        self.model.load_state_dict(torch.load(PTH))

    #--- dispatcher ---
    def solve_maze(self):
        while True:
            self.wait_image()
            img = preprocess(self._imgBGR)
            pred = self.model(torch.unsqueeze(transform(img), 0)).argmax(1)
            self.get_logger().info(f'prediction: {pred}')
            if pred == 0:
                self.move_forward()
            elif pred == 1:
                self.rotate(np.pi/2)
            elif pred == 2:
                self.rotate(-np.pi/2)
            elif pred == 3 or pred == 4:
                self.rotate(np.pi)
            else:
                return

    #--- atomic move functions ---
    def move_forward(self):
        while True:
            self.wait_odom()

            if self.new_lidar:
                self.new_lidar = False
                if self.obst_dist <= D_TOL:
                    self._vel_publisher.publish(Twist())
                    return

            cmd = Twist()
            cmd.linear.x = MAX_VEL
            cmd.angular.z = np.clip(-KW*self.a, -MAX_ANG, MAX_ANG)
            self._vel_publisher.publish(cmd)

    def rotate(self, angle):
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

    #--- wait for topic updates ---
    def wait_odom(self):
        self.new_odom = False
        while not self.new_odom:
            rclpy.spin_once(self)

    def wait_image(self):
        self.new_img = False
        while not self.new_img:
            rclpy.spin_once(self)

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

    def _image_callback(self, CompressedImage):
        # The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
        self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        self.new_img = True

def main():
    rclpy.init() #init routine needed for ROS2.
    node = SolveMaze() #Create class object to be used.
    node.solve_maze()
    rclpy.shutdown()


if __name__ == '__main__':
    main()