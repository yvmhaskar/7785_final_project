#!/usr/bin/env python

# ROS
import rospy
import rospkg

# numpy
import numpy as np

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

# OpenCV
import cv2

def train_classifier():
    """Train a classifier and save it as a .pkl for later use."""
    rospy.init_node('train_classifier')

    # Specify the folder path and CSV file path
    images_folder_path = '/path/to/your/images/folder/'
    labels_csv_path = '/path/to/your/S2023Labels.csv'

    split = 0.4

    data, label = load_data(images_folder_path, labels_csv_path)
    print('\nImported', data.shape[0], 'training instances')

    data, label = shuffle(data, label)
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=split)

    print('Training classifier...')

    # Feature extraction using color histogram
    features_train = [extract_color_histogram(image) for image in data_train]
    features_test = [extract_color_histogram(image) for image in data_test]

    # Convert label strings to numerical values
    label_encoder = LabelEncoder()
    label_train_encoded = label_encoder.fit_transform(label_train)
    label_test_encoded = label_encoder.transform(label_test)

    # Initialize the classifier (RandomForestClassifier in this case)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    classifier.fit(features_train, label_train_encoded)

    print('Detailed results on a %0.0f/%0.0f train/test split:' % ((1 - split) * 100, split * 100))
    predicted = classifier.predict(features_test)
    print(metrics.classification_report(label_test_encoded, predicted))
    print(metrics.confusion_matrix(label_test_encoded, predicted))

    print('Training and saving a model on the full dataset...')
    classifier.fit(features_train, label_train_encoded)

    # Save the trained model
    joblib.dump(classifier, 'classifier.pkl')
    print('Saved model classifier.pkl to the current directory.')

def load_data(images_folder_path, labels_csv_path):
    """Load images from a folder and labels from a CSV file."""
    labels_data = np.loadtxt(labels_csv_path, delimiter=',', dtype=str)
    image_paths = [images_folder_path + f'{number}.png' for number in labels_data[:, 0]]

    images = [cv2.imread(image_path) for image_path in image_paths]
    labels = labels_data[:, 1]

    return np.array(images), labels

# Assuming pixel values are in the range [0, 255]
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist /= np.sum(hist)  # Normalize
    return hist

if __name__ == '__main__':
    try:
        train_classifier()
    except rospy.ROSInterruptException:
        pass
