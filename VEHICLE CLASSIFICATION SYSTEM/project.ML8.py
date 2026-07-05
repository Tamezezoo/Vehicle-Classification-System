import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. System Configurations & Constants
# ==========================================
IMG_SIZE = 128
dataset_path = "dataset"

# Fixed ordering: Cars mapped to label 0, Trucks mapped to label 1
CATEGORIES = ["Cars", "Trucks"]
print(f"Categories detected: {CATEGORIES}")

def process_image_pipeline(img_path, save_debug_img=False, debug_name=""):
    """
    Image processing pipeline: Reads the image, normalizes lighting/contrast,
    and extracts HOG features required for model training.
    """
    img = cv2.imread(img_path)
    if img is None: 
        return None
    
    # Resize image and convert to grayscale for uniformity
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast and stabilize variations in lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Extract HOG features directly from the enhanced grayscale image for better accuracy
    features = hog(enhanced_gray, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), visualize=False)
    
    # Canny edge detection is used here strictly for debugging and visualization purposes
    if save_debug_img:
        segmented = cv2.Canny(gray, 100, 200)
        cv2.imwrite(f"segmented_{debug_name}.jpg", segmented)
        
    return features

# ==========================================
# 2. Execution Pipeline
# ==========================================
print("==========================================")
print("       VEHICLE CLASSIFICATION SYSTEM      ")
print("==========================================")

# ------------------------------------------
# Step 1: Data Loading & Preprocessing
# ------------------------------------------
print("Step 1: Loading Training Dataset...")
X, y = [], []
for label, category in enumerate(CATEGORIES):
    print(f"  - Processing directory: {category} (Label ID: {label})")
    path = os.path.join(dataset_path, category)
    
    for img_name in os.listdir(path):
        features = process_image_pipeline(os.path.join(path, img_name))
        if features is not None:
            X.append(features)
            y.append(label)

X, y = np.array(X), np.array(y)
print(f"Dataset Feature Matrix Shape: {X.shape}")

print("Step 2 & 3: Preprocessing & Segmentation (Handled dynamically during load)")
print("Step 4: HOG Feature Extraction (Handled dynamically during load)")

# ------------------------------------------
# Step 2: SVM Model Training
# ------------------------------------------
print("Step 5: Training SVM Classifier...")
# 80/20 train-test split with a fixed random state for reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
print("Model trained successfully.")

# ------------------------------------------
# Step 3: Model Evaluation
# ------------------------------------------
print("\nStep 6: Evaluating Model Performance")
y_pred = model.predict(X_test)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0))

# ------------------------------------------
# Step 4: Inference on New Test Image
# ------------------------------------------
print("Step 7: Testing Model on External Image: test.jpg")
test_img_path = "test.jpg"

if os.path.exists(test_img_path):
    # Process test image and enable debug image generation
    test_features = process_image_pipeline(test_img_path, save_debug_img=True, debug_name="test_output")
    prediction = model.predict(test_features.reshape(1, -1))
    
    print(f"DEBUG: Predicted class index is: {prediction[0]}")
    
    # Retrieve the string label from the prediction index
    pred_index = prediction[0]
    predicted_label = CATEGORIES[pred_index] 
    
    # Clear any active display windows to free up resources
    cv2.destroyAllWindows() 
    
    # Resize original image and edge map to a uniform size for side-by-side display
    disp = cv2.resize(cv2.imread(test_img_path), (400, 400))
    seg_raw = cv2.imread("segmented_test_output.jpg")
    
    if seg_raw is None:
        seg = np.zeros((400, 400), dtype=np.uint8)
    else:
        seg = cv2.resize(seg_raw, (400, 400))
        if len(seg.shape) == 3:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale edge map to BGR to allow merging with the colored image
    seg_bgr = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    
    # Set text color dynamically: Red for Trucks, Green for Cars
    color = (0, 0, 255) if predicted_label == "Trucks" else (0, 255, 0)
    
    # Overlay the prediction label onto the main image
    cv2.putText(disp, f"PRED: {predicted_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    # Horizontally stack the original image and the edge map for final visualization
    combined = np.hstack((disp, seg_bgr))
    print(f"\nFinal Prediction Result: {predicted_label}")
    print("\nOpening visualization window...")
    
    cv2.imshow("Final Project - Visualization", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Target file 'test.jpg' could not be found in the specified path!")
