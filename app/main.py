from fastapi import FastAPI, File, UploadFile, HTTPException
from app.model import predict_image

app = FastAPI(
    title="CNN Image Classification API",
    description="API for image classification using a Convolutional Neural Network.",
    version="2.0.0"
)

@app.get("/")
def read_root():
    return {"status": "CNN Image Classification API is running."}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    
    return {
        "filename": file.filename,
        "predicted_class": result["class_name"],
        "confidence": result["confidence"]
    }