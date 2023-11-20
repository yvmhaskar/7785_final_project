# Joseph Sommer and Yash Mhaskar
from __future__ import print_function
import cv2 as cv
import argparse
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

def on_low_H_thresh_trackbar(val):
 global low_H
 global high_H
 low_H = val
 low_H = min(high_H-1, low_H)
 cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
 global low_H
 global high_H
 high_H = val
 high_H = max(high_H, low_H+1)
 cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
 global low_S
 global high_S
 low_S = val
 low_S = min(high_S-1, low_S)
 cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
 global low_S
 global high_S
 high_S = val
 high_S = max(high_S, low_S+1)
 cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
 global low_V
 global high_V
 low_V = val
 low_V = min(high_V-1, low_V)
 cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
 global low_V
 global high_V
 high_V = val
 high_V = max(high_V, low_V+1)
 cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()

cap = cv.VideoCapture(args.camera)
low_H = 85
low_S = 153
low_V = 76
high_H = 146

# Set adjustable trackbar for filtering
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

while True:
 ret, frame = cap.read()
 if frame is None:
    break
 blur = cv.GaussianBlur(frame, (15, 15), 0)
 frame_HSV = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
 frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
 # finding the contours
 contours, _ = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
 # take the first contour
 count = 0
 # Prevents error if nothing is detected
 if len(contours)!=0:
    count = contours[0]
    (x_axis,y_axis),radius = cv.minEnclosingCircle(count)
    center = (int(x_axis),int(y_axis))
    radius = int(radius)
 # reduces likelihood of showing contour on wrong object
 if radius>50:
     cv.circle(frame,center,radius,(0,255,0),2)
     cv.circle(frame_threshold,center,radius,(0,255,0),2)

 #print center point of object
 print("Center: %2d, %2d" % (x_axis, y_axis))
 
 cv.imshow(window_capture_name, frame)
 cv.imshow(window_detection_name, frame_threshold)
 key = cv.waitKey(30)
 if key == ord('q') or key == 27:
    break