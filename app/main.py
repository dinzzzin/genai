from fastapi import FastAPI, Response
from app.model import generate_diffusion_image, generate_ebm_image

app = FastAPI(
    title="Generative Models API (Diffusion & EBM)",
    description="CIFAR-10 Image Generation using Diffusion and Energy-Based Models",
    version="2.0.0"
)

@app.get("/")
def root():
    return {"status": "Diffusion and Energy-Based Model API is running"}

@app.post("/generate/diffusion")
def generate_diffusion():
    """Generate a 32x32 RGB image using the Diffusion Model."""
    image_bytes = generate_diffusion_image()
    return Response(content=image_bytes, media_type="image/png")

@app.post("/generate/ebm")
def generate_ebm():
    """Generate a 32x32 RGB image using the Energy-Based Model (Langevin Sampling)."""
    image_bytes = generate_ebm_image()
    return Response(content=image_bytes, media_type="image/png")