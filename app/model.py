import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Load pre-trained MobileNetV2 for fast, lightweight inference
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

# Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet labels
categories = models.MobileNet_V2_Weights.DEFAULT.meta["categories"]

def predict_image(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        
    class_name = categories[top_catid[0].item()]
    confidence = top_prob[0].item()
    
    return {"class_name": class_name, "confidence": float(confidence)}
