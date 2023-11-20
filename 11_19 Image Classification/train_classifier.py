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

# OpenCV
import cv2

def train_classifier():
    """Train a classifier and save it as a .pkl for later use."""
    #rclpy.init('train_classifier')

    #filepath = rclpy.get_param('~file_name', 'S2023Labels.csv')
    images_filepath = '/home/michaelangelo/Documents/Team19_image_classifier-master/2023Simgs/S2023_imgs/'
    labels_filepath = '/home/michaelangelo/Documents/Team19_image_classifier-master/2023Simgs/S2023Labels.csv'
    #if len(filepath) > 0 and filepath[0] != '/':
    #    rospack = rospkg.RosPack()
    #    filepath = rospack.get_path('image_classifier') + '/data/training/' + filepath

    split = 0.4

    #data, label = load_data(filepath)
    data, label = load_data(images_filepath, labels_filepath)
    print( '\nImported', data.shape[0], 'training instances')

    data, label = shuffle(data, label)
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=split) # randomizes the data into training and testing

    print( 'Training classifier...')

    ##########################################################################
    # Begin classifier initialization code (You write this!)

    # Initialize the classifier you want to train with the parameters you want here:
    # Install scikit-learn with these instructions: http://scikit-learn.org/stable/install.html
    # Models, documentation, instructions, and examples can be found here:
    #   http://scikit-learn.org/stable/supervised_learning.html#supervised-learning

    #classifier = None  # TODO: Replace this with the classifier you want



    # Feature extraction using color histogram
    features_train = [extract_color_histogram(image).reshape(-1) for image in data_train]
    print("Shape of a feature vector:", features_train[0].shape)
    features_test = [extract_color_histogram(image).reshape(-1) for image in data_test]

    # Convert label strings to numerical values
    label_encoder = LabelEncoder()
    label_train_encoded = label_encoder.fit_transform(label_train)
    label_test_encoded = label_encoder.transform(label_test)

    # Initialize the classifier (RandomForestClassifier in this case)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # End image processing code (You write this!)
    ############################################################################################

    print(data_train)
    classifier.fit(data_train, label_train)

    print('Detailed results on a %0.0f/%0.0f train/test split:' % ((1 - split)*100, split*100))
    predicted = classifier.predict(data_test)
    print(metrics.classification_report(label_test, predicted))
    print(metrics.confusion_matrix(label_test, predicted))

    print('Training and saving a model on the full dataset...')
    classifier.fit(data, label)

    joblib.dump(classifier, 'classifier.pkl')
    print('Saved model classifier.pkl to current directory.')

# Assuming pixel values are in the range [0, 255]
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist /= np.sum(hist)  # Normalize
    return hist

def load_data(images_filepath, labels_filepath):
    """Parse training data and labels from a .csv file."""
    labels_data = np.loadtxt(labels_filepath, delimiter=',', dtype=str)
    images_paths = [images_filepath + f'{number}.png' for number in labels_data[:,0]] #np.loadtxt(filepath, delimiter=',')
    #x = data[:, :data.shape[1] - 1]
    #y = data[:, data.shape[1] - 1]
    images=[cv2.imread(image_path) for image_path in images_paths]
    labels = labels_data[:,1]
    return np.array(images), labels#x, y


if __name__ == '__main__':
    try:
        train_classifier()
    except rclpy.ROSInterruptException:
        pass
