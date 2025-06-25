import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from openai import OpenAI
import os
from dotenv import load_dotenv

# .env 読み込み & OpenAI APIキー設定
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🧠 アドバイス生成
def generate_advice(label: str, reason: str) -> str:
    prompt = f"""
    この服装は「{label}」と判断されました。理由は「{reason}」です。
    さらにおしゃれにするにはどうすればよいか、1文で具体的なアドバイスをください。
    """
    print("📤 OpenAI prompt (advice):", prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        advice = response.choices[0].message.content.strip()
        print("✅ OpenAI advice:", advice)
        return advice
    except Exception as e:
        print("❌ OpenAI API error (advice):", e)
        return "アドバイス生成に失敗しました。"

# ✏️ 理由文生成（自然な表現に）
def generate_reason(label: str, position: str, color_desc: str) -> str:
    prompt = f"""
    ファッション画像の分類結果として、「{label}」と判定されました。
    注目すべきポイントは「{position}の{color_desc}」です。
    これらの要素に基づいて、自然な日本語で1文の理由を生成してください。
    """
    print("📤 OpenAI prompt (reason):", prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        reason = response.choices[0].message.content.strip()
        print("✅ OpenAI reason:", reason)
        return reason
    except Exception as e:
        print("❌ OpenAI API error (reason):", e)
        return f"{position}の{color_desc}に注目し、{label}と判断しました。"

# 📷 Grad-CAM + ラベル・特徴抽出 + GPTによる理由＆アドバイス生成
def generate_explanation(model, img_path, last_conv_layer_name="Conv_1"):
    img_size = (224, 224)

    # 画像読み込みと前処理
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Grad-CAM モデル構築
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # ヒートマップ生成
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam + 1e-8)
    cam = cv2.resize(cam, img_size)

    # 注目位置の推定
    heatmap_thresh = np.where(cam > 0.5, 1, 0).astype(np.uint8)
    moments = cv2.moments(heatmap_thresh)
    if moments["m00"] != 0:
        cy = int(moments["m01"] / moments["m00"])
    else:
        cy = img_size[1] // 2

    if cy < img_size[1] * 0.4:
        position = "上半身"
    elif cy > img_size[1] * 0.6:
        position = "下半身"
    else:
        position = "全体"

    # HSVによる色特徴量（Grad-CAM領域にマスク）
    img_cv = cv2.imread(img_path)
    img_resized = cv2.resize(img_cv, img_size)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    mask = (cam > 0.5).astype(np.uint8)
    masked_hsv = hsv * np.expand_dims(mask, axis=2)

    hue_vals = masked_hsv[..., 0][mask == 1]
    sat_vals = masked_hsv[..., 1][mask == 1]

    hue_mean = hue_vals.mean() if hue_vals.size > 0 else 0
    sat_mean = sat_vals.mean() if sat_vals.size > 0 else 0

    if sat_mean > 100:
        if hue_mean < 30 or hue_mean > 150:
            color_desc = "暖色系で鮮やかな色味"
        else:
            color_desc = "寒色系で鮮やかな色味"
    else:
        color_desc = "落ち着いた色味"

    # ラベル決定
    class_label = "おしゃれ着" if int(pred_index) == 1 else "ダル着"

    # OpenAIで自然な理由文生成
    reason = generate_reason(class_label, position, color_desc)

    # OpenAIでアドバイス生成（おしゃれ着ならスキップ）
    if class_label == "おしゃれ着":
        advice = "特に改善点はありません。今のままで十分おしゃれです！"
    else:
        advice = generate_advice(class_label, reason)

    return class_label, reason, advice