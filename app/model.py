import torch
import torch.nn as nn
from PIL import Image
import io

# -------------------------------------------------------------
# 1. Generator Architecture
# -------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        # FC layer: 100 -> 7 * 7 * 128 (6272)
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        
        # Conv Transpose 1: 128 -> 64, kernel 4, stride 2, padding 1 => 14x14
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Conv Transpose 2: 64 -> 1, kernel 4, stride 2, padding 1 => 28x28
        self.conv_trans2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)  # Reshape to (BATCH_SIZE, 128, 7, 7)
        x = self.relu(self.bn1(self.conv_trans1(x)))
        x = self.tanh(self.conv_trans2(x))
        return x

# -------------------------------------------------------------
# 2. Discriminator Architecture
# -------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Conv 1: 1 -> 64, kernel 4, stride 2, padding 1 => 14x14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)
        
        # Conv 2: 64 -> 128, kernel 4, stride 2, padding 1 => 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # FC Layer: Flatten 128 * 7 * 7 -> 1 (Probability output)
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.sigmoid(self.fc(x))
        return x

# Instantiate Generator
generator = Generator(latent_dim=100)
generator.eval()

def generate_digit_image() -> bytes:
    """Generates a random hand-written digit image using the GAN generator."""
    # Sample random noise vector of shape (1, 100)
    z = torch.randn(1, 100)
    
    with torch.no_grad():
        generated_tensor = generator(z)
        
    # Convert output tensor from [-1, 1] (Tanh) to [0, 255] byte format
    img_tensor = (generated_tensor.squeeze(0).squeeze(0) + 1) / 2.0
    img_tensor = torch.clamp(img_tensor * 255, 0, 255).byte()
    
    # Convert PyTorch Tensor to PIL Image
    pil_img = Image.fromarray(img_tensor.numpy(), mode="L")
    
    # Save to PNG byte stream
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()