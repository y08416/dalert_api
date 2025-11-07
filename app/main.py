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

# FastAPIアプリ作成
app = FastAPI()

# CORSミドルウェア設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発中は "*"、本番環境では特定のドメインに制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルのロード（起動時に一度だけ読み込む）
model = load_model("model/model.h5")

# 一時ファイル保存用フォルダ
UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 推論エンドポイント
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # 一時ファイルとして保存
    temp_filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 推論と理由・アドバイス生成
        label, reason, advice = generate_explanation(model, temp_filename)

        # 結果をJSONで返す
        return JSONResponse(content={
            "label": label,     # 判定結果（例：おしゃれ or ダル着）
            "reason": reason,   # 理由文（例：色の組み合わせが明るく統一されている）
            "advice": advice    # アドバイス（例：彩度を上げるとより印象が良くなる）
        })

    except Exception as e:
        # エラー発生時は500エラーを返す
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# ヘルスチェック（RenderやUptimeRobotなどの監視用）
@app.head("/")
def health_check():
    return {}

# 動作確認用エンドポイント
@app.get("/")
def root():
    return {"status": "ok"}

# trigger redeploy