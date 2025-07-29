from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import shutil
import os

# Initialize FastAPI app
app = FastAPI()

# Mount static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load your trained model
model = load_model("brain_tumor_classifier_vgg16.h5")

# Define class labels (update according to your dataset)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save the uploaded file to 'static/uploads'
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and preprocess image
    image = Image.open(file_location).convert("RGB")
    image = image.resize((224, 224))  # match model input size
    image_array = np.array(image) / 255.0  # normalize
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension

    # Make prediction
    prediction = model.predict(image_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    # Return result template
    return templates.TemplateResponse("result.html", {
        "request": request,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "image_path": "/" + file_location  # this makes the image visible
    })
