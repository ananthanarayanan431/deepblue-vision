import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional, List, Tuple
import math
from dataclasses import dataclass

@dataclass
class AnthropometricRatios:
    # Updated ratios based on larger dataset averages
    AVERAGE_RATIOS = {
        'ipd_to_height': 27.3,  # Updated from anthropometric studies
        'face_length_to_height': 7.5,  # Refined ratio
        'nose_to_height': 51.8,  # Adjusted based on population studies
        'eye_to_chin_to_height': 9.1,  # Updated measurement
        'bizygomatic_to_height': 13.2  # New ratio for face width
    }
    
    # Population-specific adjustment factors
    ETHNIC_ADJUSTMENTS = {
        'caucasian': 1.0,
        'asian': 0.98,
        'african': 1.02,
        'hispanic': 0.99
    }
    
    # Age-based adjustment factors
    AGE_ADJUSTMENTS = {
        'child': 1.15,      # Ages 5-12
        'teenager': 1.08,   # Ages 13-19
        'adult': 1.0,       # Ages 20-60
        'elderly': 0.98     # Ages 60+
    }

class HeightEstimator:
    def __init__(self, 
                 image_path: str,
                 ethnicity: str = 'caucasian',
                 age_group: str = 'adult',
                 gender: str = 'neutral'):
        """
        Initialize Height Estimator with enhanced parameters
        
        Args:
            image_path: Path to the image
            ethnicity: Ethnic group for ratio adjustment
            age_group: Age category for scaling
            gender: Gender for specific adjustments
        """
        self.image_path = image_path
        self.ethnicity = ethnicity
        self.age_group = age_group
        self.gender = gender
        
        # Initialize MediaPipe with improved configuration
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.8,  # Increased confidence threshold
            model_selection=1  # Using the full-range model for better accuracy
        )
        
        # Enhanced calibration factors
        self.calibration_matrix = self._initialize_calibration()

    def _initialize_calibration(self) -> np.ndarray:
        """Initialize advanced calibration matrix"""
        base_calibration = 1.02
        ethnic_factor = AnthropometricRatios.ETHNIC_ADJUSTMENTS.get(self.ethnicity, 1.0)
        age_factor = AnthropometricRatios.AGE_ADJUSTMENTS.get(self.age_group, 1.0)
        
        return np.array([
            [base_calibration * ethnic_factor * age_factor, 0, 0],
            [0, base_calibration * ethnic_factor * age_factor, 0],
            [0, 0, 1]
        ])

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced image preprocessing pipeline
        """
        # Advanced denoising with optimized parameters
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, 
            h=10,  # Luminance component
            hColor=10,  # Color components
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Adaptive histogram equalization for better contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Detail enhancement with adaptive parameters
        detail_enhanced = cv2.detailEnhance(
            enhanced,
            sigma_s=12,  # Increased for better edge preservation
            sigma_r=0.15  # Adjusted for optimal detail enhancement
        )
        
        return detail_enhanced

    def get_facial_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Enhanced facial landmark detection with confidence filtering
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return None
            
        # Get the detection with highest confidence if multiple faces are detected
        best_detection = max(results.detections, key=lambda x: x.score[0])
        
        if best_detection.score[0] < 0.8:  # Strict confidence threshold
            return None
            
        landmarks = best_detection.location_data.relative_keypoints
        image_height, image_width, _ = image.shape
        
        return [
            (int(kp.x * image_width), int(kp.y * image_height))
            for kp in landmarks
        ]

    def calculate_confidence(self, estimates: Dict[str, float], measurements: Dict[str, float]) -> float:
        """
        Enhanced confidence calculation using multiple factors
        """
        # Calculate variance-based confidence
        variance = np.var(list(estimates.values()))
        max_variance = 100
        variance_confidence = max(0, min(1, 1 - (variance / max_variance)))
        
        # Calculate measurement reliability
        measurement_scores = []
        expected_ranges = {
            'ipd': (5.5, 7.5),
            'face_length': (17, 23),
            'nose_length': (4, 6),
            'eye_to_chin': (11, 15)
        }
        
        for measure, value in measurements.items():
            if measure in expected_ranges:
                min_val, max_val = expected_ranges[measure]
                if min_val <= value <= max_val:
                    measurement_scores.append(1.0)
                else:
                    deviation = min(abs(value - min_val), abs(value - max_val))
                    measurement_scores.append(max(0, 1 - (deviation / max_val)))
        
        measurement_confidence = np.mean(measurement_scores) if measurement_scores else 0
        
        # Weighted combination of confidence scores
        final_confidence = (0.6 * variance_confidence + 0.4 * measurement_confidence)
        
        return round(final_confidence, 2)

    def estimate_height(self) -> Optional[Dict]:
        """
        Enhanced height estimation with improved accuracy
        """
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError("Could not load image")

        image = self.preprocess_image(image)
        landmarks = self.get_facial_landmarks(image)

        if landmarks is None:
            return None

        # Extract facial measurements with calibration
        measurements = self._calculate_calibrated_measurements(landmarks)
        
        # Calculate estimates using multiple methods
        estimates = self._calculate_height_estimates(measurements)
        
        # Dynamic weight calculation based on measurement reliability
        weights = self._calculate_dynamic_weights(measurements, estimates)
        
        # Weighted average with reliability factors
        estimated_height = sum(estimates[key] * weights[key] for key in estimates)
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence(estimates, measurements)

        return {
            'estimated_height': round(estimated_height, 1),
            'confidence_score': confidence_score,
            'individual_estimates': estimates,
            'measurements': measurements,
            'reliability_weights': weights
        }

    def _calculate_calibrated_measurements(self, landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
        """Calculate calibrated facial measurements"""
        scale_factor = self._calculate_scale_factor(landmarks)
        
        measurements = {
            'ipd': calculate_distance(landmarks[0], landmarks[1]) / scale_factor,
            'face_length': calculate_distance(landmarks[2], landmarks[3]) / scale_factor,
            'nose_length': calculate_distance(landmarks[2], landmarks[3]) / scale_factor,
            'eye_to_chin': calculate_distance(landmarks[1], landmarks[3]) / scale_factor
        }
        
        # Apply calibration matrix
        calibrated_measurements = {
            key: value * self.calibration_matrix[0][0]
            for key, value in measurements.items()
        }
        
        return calibrated_measurements

    def _calculate_dynamic_weights(self, measurements: Dict[str, float], 
                                 estimates: Dict[str, float]) -> Dict[str, float]:
        """Calculate dynamic weights based on measurement reliability"""
        # Base weights
        weights = {
            'ipd_based': 0.35,
            'face_length_based': 0.25,
            'nose_based': 0.20,
            'eye_chin_based': 0.20
        }
        
        # Adjust weights based on measurement reliability
        reliability_scores = self._calculate_measurement_reliability(measurements)
        
        # Normalize weights
        total_reliability = sum(reliability_scores.values())
        if total_reliability > 0:
            weights = {
                key: (weights[key] * reliability_scores[key.replace('_based', '')]) / total_reliability
                for key in weights
            }
        
        return weights

    def _calculate_measurement_reliability(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """Calculate reliability scores for each measurement"""
        reliability_scores = {}
        
        # Define expected ranges and calculate reliability based on deviation
        expected_ranges = {
            'ipd': (5.5, 7.5),
            'face_length': (17, 23),
            'nose_length': (4, 6),
            'eye_to_chin': (11, 15)
        }
        
        for measure, value in measurements.items():
            if measure in expected_ranges:
                min_val, max_val = expected_ranges[measure]
                if min_val <= value <= max_val:
                    reliability_scores[measure] = 1.0
                else:
                    deviation = min(abs(value - min_val), abs(value - max_val))
                    reliability_scores[measure] = max(0, 1 - (deviation / max_val))
        
        return reliability_scores

def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\anant\OneDrive\Desktop\DeepBlue\Backend\src\Images\image.png"
    reference_object_height = 10.0
    reference_object_points = [(100, 200), (100, 300)]

    result = HeightEstimator(image_path).estimate_height()
    if result:
        print(f"Estimated height: {result['estimated_height']:.2f} cm")
        print("\nConfidence metrics:")
        for method, height in result['confidence_metrics'].items():
            print(f"{method}: {height:.2f} cm")
    else:
        print("No face detected in the image.")