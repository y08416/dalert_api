import os
from fastapi import Request, HTTPException
from jose import jwt
from dotenv import load_dotenv

# .env を読み込む（JWT秘密鍵含む）
load_dotenv()

# SupabaseがHS256で署名したトークンを検証するためのシークレット
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
ALGORITHM = "HS256"

if not SUPABASE_JWT_SECRET:
    raise RuntimeError("SUPABASE_JWT_SECRET is not set in the environment.")

def get_current_user_id(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]

    try:
        print("🔐 Decoding HS256 token with local secret...")
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=[ALGORITHM],
            options={"verify_aud": False}  # 👈 ここでaudienceの検証を無効化
        )
        print("✅ Token valid. User ID (sub):", payload["sub"])
        return payload["sub"]
    except Exception as e:
        print("❌ Token validation failed:", repr(e))
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")