import os
from dotenv import load_dotenv
from supabase import create_client

# .env の読み込み
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET")

# クライアント初期化
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 画像アップロード
def upload_image(file_path: str, file_name: str):
    print(f"📤 Uploading {file_name} to Supabase Storage...")
    try:
        with open(file_path, "rb") as f:
            res = supabase.storage.from_(BUCKET_NAME).upload(file_name, f, {"content-type": "image/jpeg"})
        print("✅ Upload successful:", res)
        return res
    except Exception as e:
        print("❌ Upload failed:", e)
        return None

# 公開URLの取得
def get_public_url(file_name: str):
    try:
        res = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
        url = res["publicUrl"] if isinstance(res, dict) else res
        print(f"🌐 Public URL: {url}")
        return url
    except Exception as e:
        print("❌ Failed to get public URL:", e)
        return None

# predictions テーブルに記録
def insert_prediction(user_id: str, image_url: str, result: str, reason: str):
    print("📝 Saving prediction to DB...")
    try:
        res = supabase.table("predictions").insert({
            "user_id": user_id,
            "image_url": image_url,
            "result": result,
            "reason": reason
        }).execute()
        print("✅ DB insert success:", res)
        return res
    except Exception as e:
        print("❌ DB insert failed:", e)
        return None