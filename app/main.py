from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os, uuid, shutil

from app.model_loader import load_model
from app.explain import generate_explanation, generate_advice

load_dotenv()
app = FastAPI()

model = load_model("model/model.h5")
UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    temp_filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        label, reason = generate_explanation(model, temp_filename)
        advice = generate_advice(label, reason)
        return JSONResponse(content={
            "label": label,
            "reason": reason,
            "advice": advice
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})