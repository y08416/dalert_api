from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from .model_loader import load_model
from .explain import generate_explanation
import shutil
import uuid
import os

app = FastAPI()

# グローバルでモデルを1回だけ読み込む
model = load_model("model/model.h5")

UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # 一時ファイルとして保存
    temp_filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        label, reason = generate_explanation(model, temp_filename)
        return JSONResponse(content={"label": label, "reason": reason})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(temp_filename)

# GET リクエスト用ヘルスチェック
@app.get("/")
def health_check():
    return {"status": "ok"}

# HEAD リクエストにも明示的に対応（Uptime Robot対策）
@app.head("/")
def health_check_head():
    return {}