from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn

app = FastAPI()

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    # Process uploaded files
    return {"files_uploaded": [file.filename for file in files]}

@app.post("/predict/")
async def predict_logo(file: UploadFile = File(...)):
    # Make predictions with your logo detection model
    # Replace this with your actual prediction logic
    prediction = {"company": "example_company", "confidence": 0.85}
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
