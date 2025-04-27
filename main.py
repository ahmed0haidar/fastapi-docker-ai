from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

# FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (good for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()


# Load the TorchScript model
model = torch.jit.load("resnet18_fx_quantized_scripted.pt")
model.eval()

# Class names
class_names = ["Cat", "Dog"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    return {"prediction": label}
