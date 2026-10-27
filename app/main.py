from fastapi import FastAPI, Response
from app.model import generate_digit_image

app = FastAPI(
    title="GAN Digit Generator API",
    description="MNIST Hand-written Digit Generation API using PyTorch GAN",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "GAN Digit Generator API is running"}

@app.post("/generate")
def generate():
    """Endpoint to generate a new synthetic handwritten digit image."""
    image_bytes = generate_digit_image()
    return Response(content=image_bytes, media_type="image/png")