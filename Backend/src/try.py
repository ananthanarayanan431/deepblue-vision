import cv2
import mediapipe as mp
import numpy as np

# Suppress MediaPipe logs
import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to estimate height based on facial features
def estimate_height_from_face(image_path, reference_object_height, reference_object_points):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return None

    image_height, image_width, _ = image.shape

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Face Detection
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            # Get facial landmarks
            landmarks = detection.location_data.relative_keypoints

            # Extract key facial points
            right_eye = (int(landmarks[mp_face_detection.FaceKeyPoint.RIGHT_EYE].x * image_width),
                         int(landmarks[mp_face_detection.FaceKeyPoint.RIGHT_EYE].y * image_height))
            left_eye = (int(landmarks[mp_face_detection.FaceKeyPoint.LEFT_EYE].x * image_width),
                        int(landmarks[mp_face_detection.FaceKeyPoint.LEFT_EYE].y * image_height))
            nose_tip = (int(landmarks[mp_face_detection.FaceKeyPoint.NOSE_TIP].x * image_width),
                        int(landmarks[mp_face_detection.FaceKeyPoint.NOSE_TIP].y * image_height))
            mouth_center = (int(landmarks[mp_face_detection.FaceKeyPoint.MOUTH_CENTER].x * image_width),
                           int(landmarks[mp_face_detection.FaceKeyPoint.MOUTH_CENTER].y * image_height))

            # Calculate interpupillary distance (distance between eyes) in pixels
            interpupillary_distance_px = calculate_distance(right_eye, left_eye)

            # Calculate face length in pixels
            top_of_head = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2 - interpupillary_distance_px)
            chin = mouth_center
            face_length_px = calculate_distance(top_of_head, chin)

            # Calculate the scale (pixels per cm) using the reference object
            reference_pixel_length = calculate_distance(reference_object_points[0], reference_object_points[1])
            scale = reference_pixel_length / reference_object_height  # pixels per cm

            # Convert facial measurements to real-world units (cm)
            interpupillary_distance_cm = interpupillary_distance_px / scale
            face_length_cm = face_length_px / scale

            # Use calibrated anthropometric ratios to estimate height
            # Example ratios (calibrated for a person with height 175 cm):
            # - Height ≈ 26.92 * interpupillary distance
            # - Height ≈ 7.5 * face length
            height_from_eyes = interpupillary_distance_cm * 26.92
            height_from_face_length = face_length_cm * 7.5

            # Average the estimates for better accuracy
            estimated_height = (height_from_eyes + height_from_face_length) / 2

            return estimated_height
    else:
        return None

# Example usage
image_path = r"C:\Users\anant\OneDrive\Desktop\DeepBlue\Backend\src\Images\virat.png"  # Replace with your selfie image
reference_object_height = 10.0  # Height of the reference object in cm
reference_object_points = [(100, 200), (100, 300)]  # Pixel coordinates of the reference object's top and bottom

estimated_height = estimate_height_from_face(image_path, reference_object_height, reference_object_points)
if estimated_height:
    print(f"Estimated height: {estimated_height:.2f} cm")
else:
    print("No face detected in the image.")