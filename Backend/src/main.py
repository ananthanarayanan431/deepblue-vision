# import os
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from dotenv import load_dotenv
# load_dotenv()

# from PIL import Image
# from phi.agent.agent import Agent
# from phi.model.google.gemini import Gemini
# from phi.model.openai import OpenAIChat
# # from phi.tools.duckducukgo import DuckDuckGo
# import uvicorn
# import io
# import base64

# app = FastAPI(
#     title="Physical Attribute and Dimension Analysis Tool",
#     description="AI-powered image analysis for predicting physical attributes and object dimensions"
# )

# # Add CORS middleware to allow cross-origin requests
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# # )

# # # Create AI agent
# medical_agent = Agent(
#     model=Gemini(
#         api_key=os.environ['GOOGLE_API_KEY'],
#         id="gemini-2.0-flash-exp"
#     ),
#     markdown=True
# )

# image_agent = Agent(
#     model=OpenAIChat(id="gpt-4o"),
#     markdown=True
# )

# # Analysis query template
# ANALYSIS_QUERY = """
# You are a highly skilled expert in computer vision and image processing, specializing in analyzing physical attributes and object dimensions. 
# Evaluate the provided image and structure your response as follows:

# ### 1. Image Type & Context
# - Specify the type of image (selfie/object/environmental image).
# - Identify the subject of analysis (e.g., human, object, or mixed)
# - Comment on image quality, resolution, and suitability for measurement tasks.

# ### 2. Physical Attributes & Dimensions Analysis
# - Extract key physical attributes (e.g., height, weight, age, body dimensions) for human subjects or say Not possible
# - Predict the height of the person in the image
# - Predict object dimensions (e.g., height, width, depth) for non-human subjects.
# - Provide measurements in consistent units with confidence intervals

# ### 3. Detailed Analysis
# - Identify specific features detected in the image
# - Suggest improvements to the current algorithm for better measurement accuracy

# ### 4. Subject Classification
# - Indicate whether the subject is human, object, or mixed
# - Provide evidence supporting the classification

# ### 5. User-Friendly Explanation
# - Present findings in simple terms
# - Explain how the measurements were derived and their accuracy
# """

# @app.post("/analyze-image/")
# async def analyze_image(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
        
#         with open("temp_image.png", "wb") as buffer:
#             buffer.write(contents)
        
  
#         response = image_agent.run(ANALYSIS_QUERY, images=["temp_image.png"])

#         with Image.open("temp_image.png") as img:
#             width, height = img.size
        
#         base64_image = base64.b64encode(contents).decode('utf-8')

#         os.remove("temp_image.png")
        
#         return JSONResponse(content={
#             "analysis": response.content,
#             "imageUrl": f"data:image/png;base64,{base64_image}",
#             "dimensions": {
#                 "width": width,
#                 "height": height
#             },
#             "status": "success"
#         })
    
#     except Exception as e:
#         if os.path.exists("temp_image.png"):
#             os.remove("temp_image.png")
        
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {
#         "message": "Physical Attribute and Dimension Analysis Tool",
#         "endpoints": {
#             "analyze_image": "/analyze-image/"
#         }
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)