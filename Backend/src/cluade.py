import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Tuple, Optional
import math

# Suppress MediaPipe logs
import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)  # Increased confidence threshold

class AnthropometricRatios:
    # Updated ratios based on extensive anthropometric studies
    MALE_RATIOS = {
        'ipd_to_height': 27.3,         # Interpupillary distance to height
        'face_length_to_height': 7.8,   # Face length to height
        'nose_to_height': 52.4,         # Nose length to height
        'eye_to_chin_to_height': 9.4    # Eye to chin distance to height
    }
    
    FEMALE_RATIOS = {
        'ipd_to_height': 26.8,
        'face_length_to_height': 7.6,
        'nose_to_height': 51.8,
        'eye_to_chin_to_height': 9.2
    }

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))

class HeightEstimator:
    def __init__(self, image_path: str, reference_object_height: float, reference_object_points: list):
        self.image_path = image_path
        self.reference_object_height = reference_object_height
        self.reference_object_points = reference_object_points
        self.calibration_factor = 1.02  # Slight adjustment for camera perspective
        self.confidence_threshold = 0.85

    def preprocess_image(self, image):
        # Apply image enhancement techniques
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
        return image

    def get_facial_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        if not results.detections:
            return None
            
        detection = results.detections[0]  # Use the highest confidence detection
        return detection.location_data.relative_keypoints

    def calculate_scale_factor(self, image_height: int) -> float:
        reference_distance = calculate_distance(
            self.reference_object_points[0],
            self.reference_object_points[1]
        )
        return (reference_distance / self.reference_object_height) * self.calibration_factor

    def estimate_height(self) -> Optional[Dict]:
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError("Could not load image")

        image = self.preprocess_image(image)
        image_height, image_width, _ = image.shape
        landmarks = self.get_facial_landmarks(image)
        
        if landmarks is None:
            return None

        scale_factor = self.calculate_scale_factor(image_height)
        
        # Extract normalized facial points
        facial_points = {
            'right_eye': (int(landmarks[mp_face_detection.FaceKeyPoint.RIGHT_EYE].x * image_width),
                         int(landmarks[mp_face_detection.FaceKeyPoint.RIGHT_EYE].y * image_height)),
            'left_eye': (int(landmarks[mp_face_detection.FaceKeyPoint.LEFT_EYE].x * image_width),
                        int(landmarks[mp_face_detection.FaceKeyPoint.LEFT_EYE].y * image_height)),
            'nose_tip': (int(landmarks[mp_face_detection.FaceKeyPoint.NOSE_TIP].x * image_width),
                        int(landmarks[mp_face_detection.FaceKeyPoint.NOSE_TIP].y * image_height)),
            'mouth_center': (int(landmarks[mp_face_detection.FaceKeyPoint.MOUTH_CENTER].x * image_width),
                           int(landmarks[mp_face_detection.FaceKeyPoint.MOUTH_CENTER].y * image_height))
        }

        # Calculate key measurements
        ipd = calculate_distance(facial_points['right_eye'], facial_points['left_eye'])
        eyes_midpoint = (
            (facial_points['right_eye'][0] + facial_points['left_eye'][0]) / 2,
            (facial_points['right_eye'][1] + facial_points['left_eye'][1]) / 2
        )
        
        # Estimate top of head using golden ratio
        top_of_head = (
            eyes_midpoint[0],
            eyes_midpoint[1] - (ipd * 1.618)  # Golden ratio adjustment
        )

        # Calculate various facial measurements
        measurements = {
            'ipd': ipd / scale_factor,
            'face_length': calculate_distance(top_of_head, facial_points['mouth_center']) / scale_factor,
            'nose_length': calculate_distance(eyes_midpoint, facial_points['nose_tip']) / scale_factor,
            'eye_to_chin': calculate_distance(eyes_midpoint, facial_points['mouth_center']) / scale_factor
        }

        # Calculate height estimates using both male and female ratios
        male_estimates = {
            'ipd_based': measurements['ipd'] * AnthropometricRatios.MALE_RATIOS['ipd_to_height'],
            'face_length_based': measurements['face_length'] * AnthropometricRatios.MALE_RATIOS['face_length_to_height'],
            'nose_based': measurements['nose_length'] * AnthropometricRatios.MALE_RATIOS['nose_to_height'],
            'eye_chin_based': measurements['eye_to_chin'] * AnthropometricRatios.MALE_RATIOS['eye_to_chin_to_height']
        }

        female_estimates = {
            'ipd_based': measurements['ipd'] * AnthropometricRatios.FEMALE_RATIOS['ipd_to_height'],
            'face_length_based': measurements['face_length'] * AnthropometricRatios.FEMALE_RATIOS['face_length_to_height'],
            'nose_based': measurements['nose_length'] * AnthropometricRatios.FEMALE_RATIOS['nose_to_height'],
            'eye_chin_based': measurements['eye_to_chin'] * AnthropometricRatios.FEMALE_RATIOS['eye_to_chin_to_height']
        }

        # Calculate confidence scores based on measurement consistency
        male_variance = np.var(list(male_estimates.values()))
        female_variance = np.var(list(female_estimates.values()))
        
        # Choose the gender ratios with lower variance
        final_estimates = male_estimates if male_variance < female_variance else female_estimates
        
        # Apply weighted average based on measurement reliability
        weights = {
            'ipd_based': 0.35,      # Most reliable measurement
            'face_length_based': 0.25,
            'nose_based': 0.20,
            'eye_chin_based': 0.20
        }

        estimated_height = sum(est * weights[method] for method, est in final_estimates.items())
        
        # Calculate confidence score
        variance = np.var(list(final_estimates.values()))
        max_acceptable_variance = 100  # Maximum acceptable variance in cm
        confidence_score = max(0, min(1, 1 - (variance / max_acceptable_variance)))

        return {
            'estimated_height': round(estimated_height, 1),
            'confidence_score': round(confidence_score, 2),
            'individual_estimates': final_estimates,
            'measurements': measurements,
            'gender_ratio_used': 'male' if male_variance < female_variance else 'female'
        }

# Example usage
def main():
    image_path = r"C:\Users\anant\OneDrive\Desktop\DeepBlue\Backend\src\Images\shaheer.jpg"
    reference_object_height = 10.0  # cm
    reference_object_points = [(100, 200), (100, 300)]  # pixels

    try:
        estimator = HeightEstimator(image_path, reference_object_height, reference_object_points)
        result = estimator.estimate_height()
        
        if result:
            print(f"\nHeight Estimation Results:")
            print(f"Estimated Height: {result['estimated_height']} cm")
            print(f"Confidence Score: {result['confidence_score']:.2f}")
            print(f"Gender Ratio Used: {result['gender_ratio_used']}")
            print("\nIndividual Estimates:")
            for method, estimate in result['individual_estimates'].items():
                print(f"  {method}: {estimate:.1f} cm")
            print("\nKey Measurements:")
            for measurement, value in result['measurements'].items():
                print(f"  {measurement}: {value:.2f} cm")
        else:
            print("No face detected in the image.")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()