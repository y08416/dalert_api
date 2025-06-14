from fastapi import FastAPI, File, UploadFile, Request, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import uuid
import shutil

from app.model_loader import load_model
from app.explain import generate_explanation, generate_advice
from app.supabase_client import upload_image, get_public_url, insert_prediction
from app.auth import get_current_user_id  # ← 追加

# 環境変数の読み込み (.env)
load_dotenv()

# FastAPI アプリの初期化
app = FastAPI()

# モデルの読み込み（グローバルに一度だけ）
model = load_model("model/model.h5")

# アップロード用一時フォルダの作成
UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 画像を受け取って分類・説明・アドバイスを返すエンドポイント + Supabase連携
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)  # ← JWTからUUIDを取得
):
    temp_filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 推論処理
        label, reason = generate_explanation(model, temp_filename)
        advice = generate_advice(label, reason)

        # Supabase Storageに画像アップロード
        file_name = f"uploads/{uuid.uuid4().hex}_{file.filename}"
        upload_result = upload_image(temp_filename, file_name)
        if not upload_result:
            return JSONResponse(status_code=500, content={"error": "Upload to Supabase failed"})

        # 公開URL取得
        image_url = get_public_url(file_name)
        if not image_url:
            return JSONResponse(status_code=500, content={"error": "Failed to get public URL"})

        # DB保存
        insert_result = insert_prediction(user_id, image_url, label, reason)
        if not insert_result:
            return JSONResponse(status_code=500, content={"error": "Failed to insert to DB"})

        return JSONResponse(content={
            "label": label,
            "reason": reason,
            "advice": advice,
            "image_url": image_url
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# UptimeRobot や Render のヘルスチェック用
@app.head("/")
def health_check():
    return {}