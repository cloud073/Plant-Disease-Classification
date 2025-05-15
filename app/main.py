import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .models.resnet9 import ResNet9
from .utils import load_model, get_default_device, to_device,get_transforms

app = FastAPI(title="Plant Disease Classification API",
             description="API for classifying plant diseases using ResNet9 model")

# Load model
device = get_default_device()
model = load_model(device)
transform = get_transforms()

# Class names (replace with your actual class names)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
@app.get("/")
def read_root():
    return {"message": "Plant Disease Classification API (CPU version)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and validate image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Apply transformations
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = to_device(image, device)
        
        # Predict
        with torch.no_grad():
            output = model(image)
            _, preds = torch.max(output, dim=1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
            
        # Prepare response
        predicted_class = class_names[preds[0].item()]
        confidence_score = round(confidence[preds[0]].item(), 2)
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence_score,
            "class_id": int(preds[0].item())
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
