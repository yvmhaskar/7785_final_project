import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.externals import joblib

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

def train_classifier():
    # ... (existing code)

    # Assuming you have a list of contours corresponding to each image
    contours_list = [cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for thresh in thresh_list]

    # Extract features from contours
    contour_features = [extract_contour_features(contours) for contours in contours_list]

    # Flatten the list of features
    flat_contour_features = [feature for sublist in contour_features for feature in sublist]

    # Combine contour features with other features if needed
    # Modify this part based on your specific data structure and feature extraction
    all_features = np.concatenate((data, flat_contour_features), axis=1)

    data, label = shuffle(all_features, label)
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=split)

    # ... (existing code)

    # Train and save the classifier
    classifier = RandomForestClassifier()
    classifier.fit(data_train, label_train)

    # Evaluate on the test set
    predicted = classifier.predict(data_test)
    print('Detailed results on a %0.0f/%0.0f train/test split:' % ((1 - split)*100, split*100))
    print(classification_report(label_test, predicted))
    print(confusion_matrix(label_test, predicted))

    # Save the trained model
    joblib.dump(classifier, 'classifier.pkl')
    print('Saved model classifier.pkl to the current directory.')

# ... (rest of the code)
