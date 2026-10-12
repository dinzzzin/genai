import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms as transforms

# Exact CNN architecture specified in Practice 1
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        # Input: 64x64x3
        # Conv1: 16 filters, 3x3, stride 1, padding 1 -> 64x64x16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 32x32x16
        
        # Conv2: 32 filters, 3x3, stride 1, padding 1 -> 32x32x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 16x16x32
        
        # FC Layers: 16 * 16 * 32 = 8192 features
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CIFAR-10 Classes
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Instantiate model
model = CNNClassifier(num_classes=10)
model.eval()

# Image Preprocessing matching 64x64 input requirement
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            top_prob, top_class_id = torch.topk(probabilities, 1)
            
        class_name = CLASSES[top_class_id[0].item()]
        confidence = float(top_prob[0].item())
        
        return {"predicted_class": class_name, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}