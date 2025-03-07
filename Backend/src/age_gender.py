
from deepface import DeepFace
import os 

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'


path = r"C:\Users\anant\OneDrive\Desktop\DeepBlue\Backend\src\Images\meee.png"
obj = DeepFace.analyze(img_path=path)

print(obj)

from deepface import DeepFace
import os


def analyze_image(image_path):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    result = DeepFace.analyze(img_path=image_path)
    
    return result

analysis_result = analyze_image(path)
print(analysis_result)