from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import uuid
import shutil

from app.model_loader import load_model
from app.explain import generate_explanation
from app.supabase_client import upload_image, get_public_url, insert_prediction
from app.auth import get_current_user_id  # JWTからユーザーID取得

# .env の読み込み
load_dotenv()

# FastAPI アプリ作成
app = FastAPI()

# モデルはグローバルに一度だけ読み込み
model = load_model("model/model.h5")

# 一時アップロードフォルダ
UPLOAD_DIR = "tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)  # 認証済みユーザーIDを取得
):
    # 一時ファイルとして保存
    temp_filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 推論 + 自然な理由文 + アドバイス生成
        label, reason, advice = generate_explanation(model, temp_filename)

        # Supabase Storage にアップロード
        file_name = f"uploads/{uuid.uuid4().hex}_{file.filename}"
        upload_result = upload_image(temp_filename, file_name)
        if not upload_result:
            return JSONResponse(status_code=500, content={"error": "Upload to Supabase failed"})

        # 公開URL取得
        image_url = get_public_url(file_name)
        if not image_url:
            return JSONResponse(status_code=500, content={"error": "Failed to get public URL"})

        # DBに保存
        insert_result = insert_prediction(user_id, image_url, label, reason, advice)
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
        # 一時ファイルを削除
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# RenderやUptimeRobotのヘルスチェック用
@app.head("/")
def health_check():
    return {}