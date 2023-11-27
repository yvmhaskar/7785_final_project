import cv2
import numpy as np

def apply_closing(image, kernel_size):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    closing_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing_result

low_H = 0
low_S = 99
low_V = 119
high_H = 10
high_S = 255
high_V = 236
min_size = 0.1
# to actually visualize the effect of `CHAIN_APPROX_SIMPLE`, we need a proper image
image_filepath = '/home/michaelangelo/Documents/Team19_image_classifier-master/2023Simgs/S2023_imgs/115.png'
#data, label = load_data(images_filepath, labels_filepath)
image1 = cv2.imread(image_filepath)
#height, width, channels = image1.shape
#x1,y1,x2,y2 = width/4, height/4, width-(width/4), height-(height/4)
#image1 = image1[int(y1):int(y2), int(x1):int(x2)]
blur = cv2.GaussianBlur(image1,(15,15),0)
frame_HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
frame_threshold = cv2.inRange(frame_HSV,(low_H, low_S, low_V),(high_H, high_S, high_V))
frame_threshold = apply_closing(frame_threshold, 35)
#ret, thresh1 = cv2.threshold(img_gray1, 100, 200, cv2.THRESH_BINARY)
#ret, thresh1 = cv2.adaptiveThreshold(img_gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

contours, hierarchy2 = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_size]
if len(filtered_contours)!=0:
        count = filtered_contours[0]
        (x_axis,y_axis),radius = cv2.minEnclosingCircle(count)
        center = (int(x_axis),int(y_axis))
        radius = int(radius)
		# reduces likelihood of showing contour on wrong object
        if radius>40:
            cv2.circle(image1,center,radius,(0,255,0),2)
            cv2.circle(frame_threshold,center,radius,(0,255,0),2)
            x1,y1,x2,y2 = int(center[0]-radius*1.2), int(center[1]-radius*1.2), int(center[0]+radius*1.2), int(center[1]+radius*1.2)
            print(x1)
            cropped = image1[int(y1):int(y2), int(x1):int(x2)]

#filtered_contours = [contour for i, contour in enumerate(contours2) if hierarchy2[0,i,3]==-1]
image_copy2 = image1.copy()
cv2.drawContours(frame_threshold, contours, -1, (100, 200, 0), 2, cv2.LINE_AA)
cv2.imshow('SIMPLE Approximation contours', image1)
cv2.waitKey(0)
cv2.imshow('CHAIN_APPROX_SIMPLE Point only', cropped)
image_copy3 = image1.copy()
for i, contour in enumerate(contours): # loop over one contour area
    for j, contour_point in enumerate(contour): # loop over the points
    # draw a circle on the current contour coordinate
        cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 2, cv2.LINE_AA)
# see the results
print(contours)
#print(hierarchy2)
cv2.imshow('CHAIN_APPROX_SIMPLE Point only', cropped)
cv2.waitKey(0)
cv2.imwrite('contour_point_simple.jpg', image_copy3)
cv2.destroyAllWindows()
