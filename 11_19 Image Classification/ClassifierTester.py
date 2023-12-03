# ROS
#import rclpy
#import rospkg

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
import cv2

# scikit-image
from skimage import transform, color, exposure
from skimage.feature import SIFT, hog

from scipy.stats import skew

# Function to train the classifier
def train_classifier():
    images_filepath ='/home/ymhaskar/VisionFollowing/2022Fimgs/' # TESTING ON 2022Fimgs
    labels_filepath = '/home/ymhaskar/VisionFollowing/2022Fimgs/labels2.txt'

    classifier_filepath = '/home/ymhaskar/classifier.pkl'

    # Load data
    data, label = load_data(images_filepath, labels_filepath)
    print('\nImported', data.shape[0], 'training instances')
    #print(label)
    data_test_cropped = [crop_function(image) for image in data]

    # Feature extraction using color histogram
    color_hist_features_test = [extract_color_histogram(image).reshape(-1) for image in data_test_cropped]

    # Feature extraction using HOG
    hog_features_test = [extract_hog_features(image).reshape(-1) for image in data]

    # Feature extraction using HOG skewness
    skewness_features_test = [compute_hog_skewness(image) for image in data]

    print('color_hist_features_test shape:', color_hist_features_test[0].shape)
    print('hog_features_test shape:', hog_features_test[0].shape)
    print('skewness_features_test shape:', skewness_features_test[0].shape)

    features_test = np.hstack((color_hist_features_test, hog_features_test, skewness_features_test))

    # Convert label strings to numerical values
    label_encoder = LabelEncoder()
    fitting_labels =  ['0','1','2','3','4','5','6','7']
    label_encoder.fit(fitting_labels)
    label_test_encoded = label_encoder.transform(label)

    # Initialize the classifier (RandomForestClassifier in this case)
    classifier = joblib.load(classifier_filepath)

    # Evaluate on the test set
    predicted = classifier.predict(features_test)
    print(metrics.classification_report(label_test_encoded, predicted))
    print(metrics.confusion_matrix(label_test_encoded, predicted))

def crop_function(image):
    low_H = 0
    low_S = 99
    low_V = 119
    high_H = 10
    high_S = 255
    high_V = 236
    min_size = 0.1
    no_contour = 0
    blur = cv2.GaussianBlur(image,(15,15),0)
    blur_HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(blur_HSV,(low_H, low_S, low_V),(high_H, high_S, high_V))
    frame_threshold = apply_closing(frame_threshold, 35)
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
        frame_threshold = apply_closing(frame_threshold, 35)
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
    #print(no_contour)
    #cv2.waitKey(0)
    #print(no_contour)
    return cropped

def apply_closing(image, kernel_size):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    closing_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing_result

# Function to extract color histogram features
def extract_color_histogram2(image):
    blur = cv2.GaussianBlur(image,(15,15),0)
    image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    #print('reached')
    ## check if sum is 0
    if np.sum(hist)==0:
        print('Zero sum')
        return np.zeros_like(hist)
    hist /= np.sum(hist)  # Normalize
    return hist

# Function to extract color histogram features
def extract_color_histogram(image):
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
def extract_sift_features(image):
    gray_image = color.rgb2gray(image)
    sift = SIFT()
    sift.detect_and_extract(gray_image)
    return sift.descriptors.flatten()

# Function to extract HOG features
def extract_hog_features(image):
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
def compute_hog_skewness(image):
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
def normalize_Skew(skew_vals):
    norm_vals = np.zeros(len(skew_vals))
    for i in range(0,len(skew_vals)):
        norm_vals[i] = (skew_vals[i]-min(skew_vals))/(max(skew_vals)-min(skew_vals))
    return norm_vals

# Function to load data from CSV files
def load_data(images_filepath, labels_filepath):
    labels_data = np.loadtxt(labels_filepath, delimiter=',', dtype=str)
    images_paths = [images_filepath + f'{number}.png' for number in labels_data[:, 0]] #convert jpg to png if needed
    images = [cv2.imread(image_path) for image_path in images_paths]
    labels = labels_data[:, 1]
    labels_array = np.array(labels, dtype=int).reshape(-1,1)
    #print('labels shape:', labels_array.shape)
    #print(labels_array)
    return np.array(images), labels_array

# Main entry point
if __name__ == '__main__':
    #try:
    train_classifier()
    #except rclpy.ROSInterruptException:
        #pass
