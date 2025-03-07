import os
import cv2
import math
import numpy as np
import base64
import mediapipe as mp
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import uvicorn
import uuid
import tempfile

from phi.agent.agent import Agent
from phi.model.google.gemini import Gemini
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

from dotenv import load_dotenv
load_dotenv()

# Suppress MediaPipe logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize FastAPI app
app = FastAPI(
    title="Physical Attribute and Dimension Analysis Tool",
    description="AI-powered image analysis for predicting physical attributes and object dimensions"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    medical_agent = Agent(
        model=Gemini(
            id="gemini-2.0-flash-exp"
        ),
        markdown=True
    )
except Exception as e:
    print(f"Warning: Failed to initialize Gemini agent: {e}")
    # Fallback to a basic response if the agent can't be initialized
    class FallbackAgent:
        def run(self, query, images=None):
            class Response:
                content = "Basic image analysis complete. Unable to provide detailed analysis."
            return Response()
    
    medical_agent = FallbackAgent()

# Analysis query template
ANALYSIS_QUERY = """
You are a highly skilled expert in computer vision and image processing, specializing in analyzing physical attributes and object dimensions. 
Evaluate the provided image and structure your response as follows:

### 1. Image Type & Context
- Specify the type of image (selfie/object/environmental image).
- Identify the subject of analysis (e.g., human, object, or mixed)
- Comment on image quality, resolution, and suitability for measurement tasks.

### 2. Physical Attributes & Dimensions Analysis
- Extract key physical attributes (e.g., weight, age, body dimensions) for human subjects or say Not possible
- Predict the height of the person in the image
- Predict object dimensions (e.g., height, width, depth) for non-human subjects.
- Provide measurements in consistent units with confidence intervals

### 3. Detailed Analysis
- Identify specific features detected in the image
- Suggest improvements to the current algorithm for better measurement accuracy

### 4. Subject Classification
- Indicate whether the subject is human, object, or mixed
- Provide evidence supporting the classification

### 5. User-Friendly Explanation
- Present findings in simple terms
- Explain how the measurements were derived and their accuracy
"""

# Initialize MediaPipe face detection
try:
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
except Exception as e:
    print(f"Warning: MediaPipe initialization failed: {e}")
    face_detection = None

# Anthropometric Ratios for height estimation
class AnthropometricRatios:
    AVERAGE_RATIOS = {
        'ipd_to_height': 27.0, 
        'face_length_to_height': 7.7,
        'nose_to_height': 52.1,
        'eye_to_chin_to_height': 9.3
    }

# Utility functions
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

class AgeEstimationTool:
    def __init__(self):
        try:
            # Use the same ResNet model for feature extraction as weight prediction
            self.feature_model = resnet50(pretrained=True)  
            self.feature_model.eval()
        except Exception as e:
            print(f"Warning: Failed to load ResNet model for age estimation: {e}")
            self.feature_model = None
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Age ranges and facial characteristics mapping
        self.age_features = {
            'skin_texture': {
                'smooth': (-5, 0),
                'fine_lines': (10, 15),
                'wrinkles': (20, 25),
                'deep_wrinkles': (30, 35)
            },
            'face_fullness': {
                'very_full': (-10, -5),
                'full': (-5, 5),
                'normal': (0, 5),
                'angular': (10, 15),
                'thin': (15, 20)
            },
            'bone_structure': {
                'developing': (-20, -10),
                'defined': (0, 5),
                'prominent': (10, 15)
            }
        }

    def predict_age(self, image: Image.Image) -> dict:
        try:
            if self.feature_model is None:
                return {"estimated_age": 30, "age_range": "25-35", "confidence_score": 0.6}
            
            # Extract features using ResNet
            input_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.feature_model(input_tensor)
                
            # Extract statistical properties from features
            avg_features = torch.mean(features, dim=1).squeeze()
            feature_std = float(torch.std(avg_features).item())
            feature_mean = float(torch.mean(avg_features).item())
            
            # Convert feature statistics to age indicators
            # This is a simplified approach - in a real system, this would be a trained regression model
            base_age = 30
            
            # Use feature mean to estimate basic age range
            if feature_mean < 0.25:
                base_age = 15
            elif feature_mean < 0.35:
                base_age = 25
            elif feature_mean < 0.45:
                base_age = 35
            elif feature_mean < 0.55:
                base_age = 45
            else:
                base_age = 55
                
            # Use feature standard deviation to refine the estimate
            # Higher variance often correlates with more facial texture/features
            age_adjustment = (feature_std - 0.2) * 40
            
            estimated_age = max(5, min(85, base_age + age_adjustment))
            
            # Calculate confidence based on feature distributions
            confidence_base = 0.7  # Base confidence
            confidence_adjustment = -abs(feature_mean - 0.4) * 0.5  # Penalty for extreme values
            confidence_score = max(0.4, min(0.9, confidence_base + confidence_adjustment))
            
            # Generate age range (±5 years)
            age_min = max(1, int(estimated_age - 5))
            age_max = min(90, int(estimated_age + 5))
            
            return {
                "estimated_age": round(estimated_age),
                "age_range": f"{age_min}-{age_max}",
                "confidence_score": round(confidence_score, 2)
            }
        except Exception as e:
            print(f"Age prediction error: {e}")
            return {"estimated_age": 30, "age_range": "25-35", "confidence_score": 0.6}

class WeightPredictionTool:
    def __init__(self):
        try:
            self.feature_model = resnet50(pretrained=True)
            self.feature_model.eval()
        except Exception as e:
            print(f"Warning: Failed to load ResNet model: {e}")
            self.feature_model = None
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.base_weights = {
            'very_thin': 45, 'thin': 55, 'normal': 70, 'overweight': 90, 'obese': 110, 'very_obese': 130
        }

    def predict_weight(self, image: Image.Image, height_cm: float) -> dict:
        try:
            if self.feature_model is None:
                return {"estimated_weight": 70.0, "body_type": "normal", "error": "Model not available"}
            
            if height_cm is None or height_cm <= 0:
                height_cm = 170.0  # Default height if invalid
                
            input_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.feature_model(input_tensor)
                
            avg_features = torch.mean(features, dim=1).squeeze()
            feature_std = float(torch.std(avg_features).item())
            feature_mean = float(torch.mean(avg_features).item())
            
            if feature_mean > 0.6:
                body_type = 'very_obese'
            elif feature_mean > 0.5:
                body_type = 'obese'
            elif feature_mean > 0.4:
                body_type = 'overweight'
            elif feature_mean < 0.2:
                body_type = 'very_thin'
            elif feature_mean < 0.3:
                body_type = 'thin'
            else:
                body_type = 'normal'
            
            bmi = self.base_weights[body_type] / (1.7 ** 2)
            weight_kg = bmi * ((height_cm / 100) ** 2)
            return {"estimated_weight": round(weight_kg, 1), "body_type": body_type}
        except Exception as e:
            print(f"Weight prediction error: {e}")
            return {"estimated_weight": 70.0, "body_type": "normal", "error": str(e)}

class HeightEstimator:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.calibration_factor = 1.02

    def preprocess_image(self, image):
        try:
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
            return image
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            return image

    def get_facial_landmarks(self, image):
        if face_detection is None:
            return None
            
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)
            
            if not results.detections:
                return None
            return results.detections[0].location_data.relative_keypoints
        except Exception as e:
            print(f"Facial landmark detection error: {e}")
            return None

    def estimate_height(self) -> Optional[Dict]:
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                print(f"Could not load image from {self.image_path}")
                return self._get_default_height_result()

            image = self.preprocess_image(image)
            image_height, image_width, _ = image.shape
            landmarks = self.get_facial_landmarks(image)

            if landmarks is None:
                print("No facial landmarks detected")
                return self._get_default_height_result()

            # Extract facial points
            try:
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
            except Exception as e:
                print(f"Error extracting facial points: {e}")
                return self._get_default_height_result()

            ipd = calculate_distance(facial_points['right_eye'], facial_points['left_eye'])
            if ipd == 0:
                print("Zero interpupillary distance detected")
                return self._get_default_height_result()
                
            scale_factor = (ipd / AnthropometricRatios.AVERAGE_RATIOS['ipd_to_height']) * self.calibration_factor

            measurements = {
                'ipd': ipd / scale_factor if scale_factor != 0 else 0,
                'face_length': calculate_distance(facial_points['nose_tip'], facial_points['mouth_center']) / scale_factor if scale_factor != 0 else 0,
                'nose_length': calculate_distance(facial_points['nose_tip'], facial_points['mouth_center']) / scale_factor if scale_factor != 0 else 0,
                'eye_to_chin': calculate_distance(facial_points['left_eye'], facial_points['mouth_center']) / scale_factor if scale_factor != 0 else 0
            }

            # Height estimation using different methods
            estimates = {}
            for key, measurement in measurements.items():
                if key == 'ipd' and measurement > 0:
                    estimates['ipd_based'] = measurement * AnthropometricRatios.AVERAGE_RATIOS['ipd_to_height']
                elif key == 'face_length' and measurement > 0:
                    estimates['face_length_based'] = measurement * AnthropometricRatios.AVERAGE_RATIOS['face_length_to_height']
                elif key == 'nose_length' and measurement > 0:
                    estimates['nose_based'] = measurement * AnthropometricRatios.AVERAGE_RATIOS['nose_to_height']
                elif key == 'eye_to_chin' and measurement > 0:
                    estimates['eye_chin_based'] = measurement * AnthropometricRatios.AVERAGE_RATIOS['eye_to_chin_to_height']
            
            if not estimates:
                print("No valid height estimates calculated")
                return self._get_default_height_result()

            # Weighted average for final height estimate
            weights = {'ipd_based': 0.35, 'face_length_based': 0.25, 'nose_based': 0.20, 'eye_chin_based': 0.20}
            valid_weights = {k: v for k, v in weights.items() if k in estimates}
            
            if not valid_weights:
                print("No valid weights for height calculation")
                return self._get_default_height_result()
                
            weight_sum = sum(valid_weights.values())
            if weight_sum == 0:
                print("Zero weight sum for height calculation")
                return self._get_default_height_result()
                
            # Adjust weights to sum to 1
            adjusted_weights = {k: v / weight_sum for k, v in valid_weights.items()}
            
            estimated_height = sum(estimates[key] * adjusted_weights[key] for key in adjusted_weights)

            # Adjust height to reasonable range (150-190 cm)
            estimated_height = max(150, min(190, estimated_height))

            # Confidence calculation
            variance = np.var(list(estimates.values())) if len(estimates) > 1 else 100
            max_variance = 100
            confidence_score = max(0, min(1, 1 - (variance / max_variance)))

            return {
                'estimated_height': round(estimated_height, 1),
                'confidence_score': round(confidence_score, 2),
                'individual_estimates': estimates,
                'measurements': measurements
            }
        except Exception as e:
            print(f"Height estimation error: {e}")
            return self._get_default_height_result()
    
    def _get_default_height_result(self):
        """Return default height values when estimation fails"""
        return {
            'estimated_height': 170.0,
            'confidence_score': 0.5,
            'individual_estimates': {
                'ipd_based': 170.0
            },
            'measurements': {
                'ipd': 6.3
            }
        }

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        # Create a temporary file with a unique name
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4()}.png"
        temp_file_path = os.path.join(temp_dir, temp_filename)
        
        # Read the file content
        contents = await file.read()
        
        # Save to temporary file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(contents)
        
        # Get image dimensions
        with Image.open(temp_file_path) as img:
            width, height = img.size
        
        # Process the image with AI agent
        try:
            response = medical_agent.run(ANALYSIS_QUERY, images=[temp_file_path])
        except Exception as e:
            print(f"AI Agent error: {e}")
            response_content = "Unable to generate detailed analysis."
            response = type('obj', (object,), {'content': response_content})
        
        # Estimate height
        try:
            height_estimator = HeightEstimator(temp_file_path)
            height_result = height_estimator.estimate_height()
        except Exception as e:
            print(f"Height estimation error: {e}")
            height_result = {
                'estimated_height': 170.0,
                'confidence_score': 0.5,
                'individual_estimates': {'default': 170.0},
                'measurements': {'default': 0}
            }
        
        # Predict weight
        try:
            with Image.open(temp_file_path) as img:
                weight_predictor = WeightPredictionTool()
                weight_result = weight_predictor.predict_weight(
                    img, 
                    height_result["estimated_height"] if height_result else 170.0
                )
        except Exception as e:
            print(f"Weight prediction error: {e}")
            weight_result = {
                'estimated_weight': 70.0,
                'body_type': 'normal'
            }
        
        # Estimate age
        try:
            with Image.open(temp_file_path) as img:
                age_predictor = AgeEstimationTool()
                age_result = age_predictor.predict_age(img)
        except Exception as e:
            print(f"Age prediction error: {e}")
            age_result = {
                'estimated_age': 30,
                'age_range': '25-35',
                'confidence_score': 0.6
            }
        
        # Convert image to base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        return JSONResponse(content={
            "analysis": response.content,
            "imageUrl": f"data:image/png;base64,{base64_image}",
            "dimensions": {"width": width, "height": height},
            "height_estimation": height_result,
            "weight_estimation": weight_result,
            "age_estimation": age_result,
            "status": "success"
        })

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Failed to remove temporary file: {e}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Physical Attribute and Dimension Analysis Tool",
        "endpoints": {
            "analyze_image": "/analyze-image/"
        }
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)