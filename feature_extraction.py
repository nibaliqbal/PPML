import cv2
import numpy as np
import tenseal as ts

# Function to extract features from an image using OpenCV
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")
    
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Enhance the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(blurred_image)
    
    # Save the enhanced image (for debugging purposes)
    enhanced_image_path = image_path.replace('.jpg', '_enhanced.jpg')
    cv2.imwrite(enhanced_image_path, enhanced_image)
    print(f'Enhanced fingerprint image saved as {enhanced_image_path}')
    
    # Example feature extraction using SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(enhanced_image, None)
    
    # For simplicity, take the first descriptor as the feature vector
    feature_vector = descriptors[0] if descriptors is not None else np.zeros(128)
    return feature_vector

# Function to encrypt a feature vector using TenSEAL
def encrypt_vector(vector, context):
    return ts.ckks_vector(context, vector)

# Function to compute the encrypted squared Euclidean distance
def encrypted_squared_euclidean_distance(enc_vec_a, enc_vec_b):
    diff = enc_vec_a - enc_vec_b
    squared_diff = diff * diff
    return squared_diff.sum()

# Main function
def main():
    # TenSEAL context setup
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    # Example image paths (you need to provide actual image paths)
    image_path_a = 'E:\project\image_a.jpg'
    image_path_b = 'E:\project\image_b.jpg'

    # Extract feature vectors
    feature_vector_a = extract_features(image_path_a)
    feature_vector_b = extract_features(image_path_b)

    # Encrypt feature vectors
    encrypted_vector_a = encrypt_vector(feature_vector_a.tolist(), context)
    encrypted_vector_b = encrypt_vector(feature_vector_b.tolist(), context)

    # Compute encrypted squared Euclidean distance
    encrypted_distance_squared = encrypted_squared_euclidean_distance(encrypted_vector_a, encrypted_vector_b)

    # Decrypt and decode the result
    distance_squared = encrypted_distance_squared.decrypt()
    euclidean_distance = np.sqrt(distance_squared)

    # Output the squared and actual Euclidean distances
    print(f"Encrypted Squared Euclidean Distance: {distance_squared}")
    print(f"Euclidean Distance: {euclidean_distance}")

if __name__ == "__main__":
    main()
