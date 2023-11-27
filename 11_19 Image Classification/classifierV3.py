# ROS
import rclpy
import rospkg

# numpy
import numpy as np

# scikit-learn (some common classifiers are included as examples, feel free to add your own)
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# OpenCV
import cv2

# scikit-image
from skimage import transform, color, exposure
from skimage.feature import SIFT, hog

# Function to train the classifier
def train_classifier():
    images_filepath = '/home/michaelangelo/Documents/Team19_image_classifier-master/2023Simgs/S2023_imgs/'
    labels_filepath = '/home/michaelangelo/Documents/Team19_image_classifier-master/2023Simgs/S2023Labels.csv'

    split = 0.4

    # Load data
    data, label = load_data(images_filepath, labels_filepath)
    print('\nImported', data.shape[0], 'training instances')

    # Shuffle and split data
    data, label = shuffle(data, label)
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=split)

    # Crop images
    data_train_cropped = [crop_function(image) for image in data_train]
    data_test_cropped = [crop_function(image) for image in data_test]
    
    # Feature extraction using color histogram
    color_hist_features_train = [extract_color_histogram(image).reshape(-1) for image in data_train_cropped]
    color_hist_features_test = [extract_color_histogram(image).reshape(-1) for image in data_test_cropped]

    # Feature extraction using SIFT
    sift_features_train = [extract_sift_features(image) for image in data_train]
    sift_features_test = [extract_sift_features(image) for image in data_test]

    # Feature extraction using HOG
    hog_features_train = [extract_hog_features(image).reshape(-1) for image in data_train]
    hog_features_test = [extract_hog_features(image).reshape(-1) for image in data_test]

    # Concatenate features
    features_train = np.hstack((color_hist_features_train, sift_features_train, hog_features_train))
    features_test = np.hstack((color_hist_features_test, sift_features_test, hog_features_test))

    # Convert label strings to numerical values
    label_encoder = LabelEncoder()
    label_train_encoded = label_encoder.fit_transform(label_train)
    label_test_encoded = label_encoder.transform(label_test)

    # Initialize the classifier (RandomForestClassifier in this case)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    classifier.fit(features_train, label_train_encoded)

    # Evaluate on the test set
    predicted = classifier.predict(features_test)
    print(metrics.classification_report(label_test_encoded, predicted))
    print(metrics.confusion_matrix(label_test_encoded, predicted))

    # Save the trained model
    joblib.dump(classifier, 'classifier.pkl')
    print('Saved model classifier.pkl to the current directory.')

# Function to extract color histogram features
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist /= np.sum(hist)  # Normalize
    return hist

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
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        block_norm='L2-Hys',
        multichannel=False
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled.flatten()

# Function to load data from CSV files
def load_data(images_filepath, labels_filepath):
    labels_data = np.loadtxt(labels_filepath, delimiter=',', dtype=str)
    images_paths = [images_filepath + f'{number}.png' for number in labels_data[:, 0]]
    images = [cv2.imread(image_path) for image_path in images_paths]
    labels = labels_data[:, 1]
    return np.array(images), labels

# Main entry point
if __name__ == '__main__':
    try:
        train_classifier()
    except rclpy.ROSInterruptException:
        pass
