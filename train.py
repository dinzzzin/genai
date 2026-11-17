import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from app.model import SimpleDiffusionUNet, EnergyModel

os.makedirs("weights", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")

# 1. Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
])

dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)


# =============================================================
# 2. Train Diffusion Model on CIFAR-10
# =============================================================
def train_diffusion(epochs: int = 3, timesteps: int = 100):
    print("\n--- Training Diffusion Model ---")
    model = SimpleDiffusionUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.size(0)

            # Sample random timesteps t and Gaussian noise
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(images)

            # Forward diffusion: q(x_t | x_0)
            sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alphas_cumprod[t])[:, None, None, None]
            noisy_images = sqrt_alpha_bar * images + sqrt_one_minus_alpha_bar * noise

            # Predict noise and compute MSE loss
            pred_noise = model(noisy_images, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Diffusion Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "weights/diffusion_cifar10.pth")
    print("Saved weights/diffusion_cifar10.pth")


# =============================================================
# 3. Train Energy-Based Model (Contrastive Divergence)
# =============================================================
def sample_langevin(ebm: nn.Module, x_init: torch.Tensor, steps: int = 10, step_size: float = 0.05) -> torch.Tensor:
    x = x_init.clone().detach()
    for _ in range(steps):
        x.requires_grad_(True)
        energy = ebm(x).sum()
        grad_x = torch.autograd.grad(energy, x)[0]
        with torch.no_grad():
            x = x - 0.5 * step_size * grad_x + math.sqrt(step_size) * torch.randn_like(x)
            x = torch.clamp(x, -1.0, 1.0)
        x = x.detach()
    return x


def train_ebm(epochs: int = 3):
    print("\n--- Training Energy-Based Model ---")
    ebm = EnergyModel().to(device)
    optimizer = torch.optim.Adam(ebm.parameters(), lr=1e-4)

    ebm.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            noise_init = torch.randn_like(real_images)

            # Generate low-energy fake samples via Langevin dynamics
            fake_images = sample_langevin(ebm, noise_init, steps=10)

            # Contrastive Divergence loss: push down energy of real data, push up fake data
            real_energy = ebm(real_images).mean()
            fake_energy = ebm(fake_images).mean()
            reg_loss = 0.01 * (real_energy**2 + fake_energy**2)
            loss = real_energy - fake_energy + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"EBM Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(ebm.state_dict(), "weights/ebm_cifar10.pth")
    print("Saved weights/ebm_cifar10.pth")


if __name__ == "__main__":
    train_diffusion(epochs=3)
    train_ebm(epochs=3)