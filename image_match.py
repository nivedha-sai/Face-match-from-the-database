import os
import pickle
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize models
mtcnn = MTCNN(keep_all=False)  # Detect a single face per image
model = InceptionResnetV1(pretrained='vggface2').eval()

def extract_features(img_path, model):
    img = Image.open(img_path)
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        img_embedding = model(img_cropped.unsqueeze(0))
        return img_embedding.detach().numpy().flatten()
    else:
        raise ValueError(f"No face detected in image: {img_path}")

# Set the paths
image_folder = r"C:\\NIVI\\Nivi"
feature_store_path = r"C:\\NIVI\\Nivi\\features.pkl"

# Function to extract and store features
def extract_and_store_features(image_folder, feature_store_path):
    features_dict = {}
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        try:
            # Check if the file is an image
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                features = extract_features(img_path, model)
                features_dict[img_name] = features
            else:
                print(f"Skipping non-image file: {img_name}")
        except (ValueError, UnidentifiedImageError) as e:
            print(f"Error processing image {img_name}: {str(e)}")
    with open(feature_store_path, 'wb') as f:
        pickle.dump(features_dict, f)

# Function to load features from file
def load_features(feature_store_path):
    with open(feature_store_path, 'rb') as f:
        return pickle.load(f)

# Function to compute cosine similarity
def compute_cosine_similarity(features1, features2):
    norm_feat1 = features1 / np.linalg.norm(features1)
    norm_feat2 = features2 / np.linalg.norm(features2)
    similarity_score = 1 - cosine(norm_feat1, norm_feat2)
    return similarity_score

# Function to count matching images and return matching filenames
def count_matching_images(new_image_path, stored_features, threshold=0.7):
    new_features = extract_features(new_image_path, model)
    match_count = 0
    matching_files = []
    for img_name, stored_feature in stored_features.items():
        similarity_score = compute_cosine_similarity(new_features, stored_feature)
        if similarity_score >= threshold:
            match_count += 1
            matching_files.append(img_name)
    return match_count, matching_files

# Extract and save features
extract_and_store_features(image_folder, feature_store_path)

# Load the stored features
stored_features = load_features(feature_store_path)

# Path to the new image to compare
new_image_path = "nivi1.jpg"

# Count how many times the same person appears and get the matching filenames
try:
    match_count, matching_files = count_matching_images(new_image_path, stored_features)
    print(f"The same person has appeared {match_count} times.")
    print("Matching files:", matching_files)
except ValueError as e:
    print(e)
