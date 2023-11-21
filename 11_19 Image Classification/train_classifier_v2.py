#!/usr/bin/env python

# ROS
import rclpy
import rospkg

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


# OpenCV
import cv2
def calculate_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    moments = cv2.moments(contour)
    orientation = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
    orientation = np.degrees(orientation)
    return [area, circularity, orientation]
    
def apply_closing(image, kernel_size):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    closing_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing_result

def contour_extraction(path):
    low_H = 0
    low_S = 99
    low_V = 119
    high_H = 10
    high_S = 255
    high_V = 236
    min_size = 0.1
    # to actually visualize the effect of `CHAIN_APPROX_SIMPLE`, we need a proper image
 
    #data, label = load_data(images_filepath, labels_filepath)
    image1 = cv2.imread(path)
    #height, width, channels = image1.shape
    #x1,y1,x2,y2 = width/4, height/4, width-(width/4), height-(height/4)
    #image1 = image1[int(y1):int(y2), int(x1):int(x2)]
    blur = cv2.GaussianBlur(image1,(15,15),0)
    frame_HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV,(low_H, low_S, low_V),(high_H, high_S, high_V))
    frame_threshold = apply_closing(frame_threshold, 35)

    #ret, thresh1 = cv2.threshold(img_gray1, 100, 200, cv2.THRESH_BINARY)
    #ret, thresh1 = cv2.adaptiveThreshold(img_gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours2, hierarchy2 = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours2 if cv2.contourArea(contour) > min_size]
    if len(filtered_contours)==0:
        print('olala')
        low_H = 20
        low_S = 90
        low_V = 0
        high_H = 126
        high_S = 255
        high_V = 255

        frame_threshold = cv2.inRange(frame_HSV,(low_H, low_S, low_V),(high_H, high_S, high_V))
        frame_threshold = apply_closing(frame_threshold, 35)
        contours2, hierarchy2 = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    image_copy2 = image1.copy()
    cv2.drawContours(image_copy2, contours2, -1, (100, 200, 0), 2, cv2.LINE_AA)
    cv2.imshow('SIMPLE Approximation contours', image_copy2)
    return contours2


def train_classifier():
    """Train a classifier and save it as a .pkl for later use."""
    #rclpy.init('train_classifier')

    #filepath = rclpy.get_param('~file_name', 'S2023Labels.csv')
    images_filepath = '/home/michaelangelo/Documents/Team19_image_classifier-master/2023Simgs/S2023_imgs/'
    labels_filepath = '/home/michaelangelo/Documents/Team19_image_classifier-master/2023Simgs/S2023_imgs/S2023Labels.csv'
    #if len(filepath) > 0 and filepath[0] != '/':
    #    rospack = rospkg.RosPack()
    #    filepath = rospack.get_path('image_classifier') + '/data/training/' + filepath

    split = 0.4

    #data, label = load_data(filepath)
    data, label, images_paths = load_data(images_filepath, labels_filepath)
    print( '\nImported', data.shape[0], 'training instances')
    



    data, label = shuffle(data, label)
    #data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=split) # randomizes the data into training and testing
    imagepath_train, imagepath_test, label_train, label_test = train_test_split(images_paths, label, test_size=split) # randomizes the data into training and testing
    
    contour_features_train = np.zeros((len(imagepath_train),3),dtype=float)
    contour_features_test = np.zeros((len(imagepath_test),3),dtype=float)

    for i,path in enumerate(imagepath_train):
        contours = contour_extraction(path)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours)>0:
            contour = contours[0]
            contour_features_train[i] = np.array(calculate_features(contour),dtype=float)
        else:
            contour_features_train[i] =  np.array([0,0,0],dtype=float)
    contour_features_train =  np.array(contour_features_train,dtype=float)


    #features_train = [contour_features_train(vector).reshape(-1,3) for vector in contour_features_train]
    features_train = contour_features_train

    for i,path in enumerate(imagepath_test):
        contours = contour_extraction(path)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0] if contours else None
        if len(contours)>0:
            contour = contours[0]
            contour_features_test[i] = np.array(calculate_features(contour),dtype=float)
        else:
            contour_features_test[i] = np.array([0,0,0],dtype=float)

    #features_test = [contour_features_test(vector).reshape(-1,3) for vector in contour_features_test]
    features_test = contour_features_test
    print(features_train)

    print( 'Training classifier...')

    ##########################################################################
    # Begin classifier initialization code (You write this!)

    # Initialize the classifier you want to train with the parameters you want here:
    # Install scikit-learn with these instructions: http://scikit-learn.org/stable/install.html
    # Models, documentation, instructions, and examples can be found here:
    #   http://scikit-learn.org/stable/supervised_learning.html#supervised-learning

    #classifier = None  # TODO: Replace this with the classifier you want




    # Feature extraction using color histogram
    #features_train = [extract_color_histogram(image).reshape(-1) for image in data_train]
    #print("Shape of a feature vector:", features_train[0].shape)
    #features_test = [extract_color_histogram(image).reshape(-1) for image in data_test]

    # Convert label strings to numerical values
    label_encoder = LabelEncoder()
    label_train_encoded = label_encoder.fit_transform(label_train)
    label_test_encoded = label_encoder.transform(label_test)
    # Initialize the classifier (RandomForestClassifier in this case)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # End image processing code (You write this!)
    ############################################################################################

    #print(data_train)
    classifier.fit(features_train, label_train_encoded)
    print('nope')
    print('Detailed results on a %0.0f/%0.0f train/test split:' % ((1 - split)*100, split*100))
    predicted = classifier.predict(features_test)
    print(metrics.classification_report(label_test_encoded, predicted))
    print(metrics.confusion_matrix(label_test_encoded, predicted))

    #print('Training and saving a model on the full dataset...')
    #classifier.fit(data, label)

    joblib.dump(classifier, 'classifier.pkl')
    print('Saved model classifier.pkl to current directory.')

# Assuming pixel values are in the range [0, 255]
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist /= np.sum(hist)  # Normalize
    return hist

def calculate_circularity(area, perimeter):
    return (4 * np.pi * area) / (perimeter ** 2)

def calculate_orientation(contour):
    moments = cv2.moments(contour)
    return 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])

def extract_contour_features(contours):
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = calculate_circularity(area, perimeter)
        orientation = calculate_orientation(contour) * 180 / np.pi  # Convert radians to degrees
        features.append([area, circularity, orientation])
    return np.array(features)

def find_contours(image):
    img_gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(img_gray1, 100, 200, cv2.THRESH_BINARY)
    #ret, thresh1 = cv2.adaptiveThreshold(img_gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

def load_data(images_filepath, labels_filepath):
    """Parse training data and labels from a .csv file."""
    labels_data = np.loadtxt(labels_filepath, delimiter=',', dtype=str)
    images_paths = [images_filepath + f'{number}.png' for number in labels_data[:,0]] #np.loadtxt(filepath, delimiter=',')
    #x = data[:, :data.shape[1] - 1]
    #y = data[:, data.shape[1] - 1]
    images=[cv2.imread(image_path) for image_path in images_paths]
    labels = labels_data[:,1]
    return np.array(images), labels, images_paths #x, y


if __name__ == '__main__':
    try:
        train_classifier()
    except rclpy.exceptions.ROSInterruptException:
        pass
