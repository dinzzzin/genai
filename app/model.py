import os
import math
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# =============================================================
# 1. Diffusion Model (UNet + Sinusoidal Time Embedding)
# =============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class SimpleDiffusionUNet(nn.Module):
    def __init__(self, time_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, 64),
            nn.SiLU()
        )
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bot = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)[:, :, None, None]
        h1 = F.silu(self.conv1(x))
        h2 = F.silu(self.down1(h1) + t_emb)
        h_bot = F.silu(self.bot(h2))
        u1 = F.silu(self.up1(h_bot))
        return self.out(torch.cat([u1, h1], dim=1))


# =============================================================
# 2. Energy-Based Model (EBM + Langevin Dynamics Sampling)
# =============================================================
class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Instantiate models and load trained CIFAR-10 weights
DIFFUSION_WEIGHTS = "weights/diffusion_cifar10.pth"
EBM_WEIGHTS = "weights/ebm_cifar10.pth"

diffusion_model = SimpleDiffusionUNet()
if os.path.exists(DIFFUSION_WEIGHTS):
    diffusion_model.load_state_dict(torch.load(DIFFUSION_WEIGHTS, map_location="cpu"))
diffusion_model.eval()

ebm_model = EnergyModel()
if os.path.exists(EBM_WEIGHTS):
    ebm_model.load_state_dict(torch.load(EBM_WEIGHTS, map_location="cpu"))
ebm_model.eval()


def _tensor_to_png_bytes(img_tensor: torch.Tensor) -> bytes:
    img = (img_tensor.squeeze(0).detach().cpu() + 1.0) / 2.0
    img = torch.clamp(img * 255.0, 0, 255).byte()
    img_np = img.permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(img_np, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def generate_diffusion_image(steps: int = 100) -> bytes:
    x = torch.randn(1, 3, 32, 32)
    betas = torch.linspace(1e-4, 0.02, steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    with torch.no_grad():
        for step in reversed(range(steps)):
            t = torch.tensor([step], dtype=torch.long)
            pred_noise = diffusion_model(x, t)
            alpha_t = alphas[step]
            alpha_bar_t = alphas_cumprod[step]
            beta_t = betas[step]
            noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise

    return _tensor_to_png_bytes(x)


def generate_ebm_image(steps: int = 30, step_size: float = 0.05) -> bytes:
    x = torch.randn(1, 3, 32, 32)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        energy = ebm_model(x).sum()
        grad_x = torch.autograd.grad(energy, x)[0]
        with torch.no_grad():
            x = x - 0.5 * step_size * grad_x + math.sqrt(step_size) * torch.randn_like(x)
            x = torch.clamp(x, -1.0, 1.0)

    return _tensor_to_png_bytes(x)