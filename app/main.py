from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import shutil

from app.model_loader import load_model
from app.explain import generate_explanation

# .env の読み込み
load_dotenv()

# FastAPI アプリ作成
app = FastAPI()

# CORSミドルウェア追加（開発中は "*"、本番は制限推奨）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 例: ["https://dalert-web.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルはグローバルに一度だけ読み込み
model = load_model("model/model.h5")

# 一時アップロードフォルダ
UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # 一時ファイルとして保存
    temp_filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 推論 + 理由文 + アドバイス生成
        label, reason, advice = generate_explanation(model, temp_filename)

        return JSONResponse(content={
            "label": label,
            "reason": reason,
            "advice": advice
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# RenderやUptimeRobotのヘルスチェック用
@app.head("/")
def health_check():
    return {}

@app.get("/")
def root():
    return {"status": "ok"}

# trigger redeploy