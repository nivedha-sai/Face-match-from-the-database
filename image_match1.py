import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize MTCNN and FaceNet models
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

def extract_features(img_path, model):
    # Load the image
    img = Image.open(img_path)
    # Detect and align the face
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        # Extract features using the model
        img_embedding = model(img_cropped[0].unsqueeze(0))
        return img_embedding.detach().numpy().flatten()
    else:
        raise ValueError(f"No face detected in image: {img_path}")

# Function to compute cosine similarity between two feature vectors
def compute_cosine_similarity(features1, features2):
    # Normalize the feature vectors
    norm_feat1 = features1 / np.linalg.norm(features1)
    norm_feat2 = features2 / np.linalg.norm(features2)
    # Compute cosine similarity (closer to 1 means more similar)
    similarity_score = 1 - cosine(norm_feat1, norm_feat2)
    return similarity_score

# Paths to the images to be compared
image1_path = "APJ.jpg"
image2_path = "APJ3.jpg"

# Extract features from both images
try:
    features1 = extract_features(image1_path, model)
    features2 = extract_features(image2_path, model)

    # Compute cosine similarity between the features
    similarity_score = compute_cosine_similarity(features1, features2)

    # Define a threshold for similarity (adjust as per your requirement)
    threshold = 0.7

    # Determine if the images represent the same person based on the similarity score
    if similarity_score >= threshold:
        print("Images are of the same person.")
    else:
        print("Images are of different persons.")

    print(f"Cosine Similarity score: {similarity_score}")
except ValueError as e:
    print(e)
