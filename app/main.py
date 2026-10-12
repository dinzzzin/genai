from fastapi import FastAPI, File, UploadFile, HTTPException
from app.model import predict_image

app = FastAPI(
    title="CNN Image Classification API",
    description="CIFAR-10 Image Classifier API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "CNN Image Classifier is running"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")
    
    try:
        contents = await file.read()
        result = predict_image(contents)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return {
            "filename": file.filename,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))