import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from openai import OpenAI
from PIL import Image
import os
from dotenv import load_dotenv
import pillow_heif

# HEIF/HEICサポートを有効化
pillow_heif.register_heif_opener()

# .env読み込み & OpenAIクライアント設定
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# アドバイス生成
def generate_advice(label: str, reason: str) -> str:
    prompt = f"""
    この服装は「{label}」と判断されました。理由は「{reason}」です。
    さらにおしゃれにするにはどうすればよいか、1文で具体的なアドバイスをください。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ OpenAI API error (advice):", e)
        return "アドバイス生成に失敗しました。"

# 理由文生成
def generate_reason(label: str, position: str, color_desc: str) -> str:
    definition = "おしゃれ着とは、外出やデートにも適した明るさ・彩度・シルエットの整った服装を指します。"
    prompt = f"""
    {definition}
    このファッション画像は「{label}」と判定されました。
    注目すべきポイントは「{position}の{color_desc}」です。
    これらに基づいて、自然な日本語で理由を1文で説明してください。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ OpenAI API error (reason):", e)
        return f"{position}の{color_desc}に注目し、{label}と判断しました。"

# 説明生成（Grad-CAM + 色特徴 + GPT）
def generate_explanation(model, img_path, last_conv_layer_name="Conv_1"):
    img_size = (224, 224)

    # モデル入力画像の前処理（Keras用）
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Grad-CAM 用モデル
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam + 1e-8)
    cam = cv2.resize(cam, img_size)

    # 注目位置推定（Y重心）
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

    # Pillowで読み込み → HSV変換
    pil_img = Image.open(img_path).convert("RGB")
    img_np = np.array(pil_img.resize(img_size))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    mask = (cam > 0.5).astype(np.uint8)
    masked_hsv = hsv * np.expand_dims(mask, axis=2)

    hue_vals = masked_hsv[..., 0][mask == 1]
    sat_vals = masked_hsv[..., 1][mask == 1]
    val_vals = masked_hsv[..., 2][mask == 1]

    hue_mean = hue_vals.mean() if hue_vals.size > 0 else 0
    sat_mean = sat_vals.mean() if sat_vals.size > 0 else 0
    val_mean = val_vals.mean() if val_vals.size > 0 else 0

    # 色特徴の記述
    if sat_mean > 100:
        if hue_mean < 30 or hue_mean > 150:
            color_desc = "暖色系で鮮やかな色味"
        else:
            color_desc = "寒色系で鮮やかな色味"
    else:
        color_desc = "落ち着いた色味"

    # ラベル判定
    if sat_mean < 100 and val_mean < 140:
        if position == "全体":
            class_label = "ダル着"
        else:
            class_label = "おしゃれ着"
    elif sat_mean < 80 and val_mean < 160 and position != "下半身":
        class_label = "ダル着"
    else:
        class_label = "おしゃれ着"

    # 説明生成
    reason = generate_reason(class_label, position, color_desc)
    advice = (
        "特に改善点はありません。今のままで十分おしゃれです！"
        if class_label == "おしゃれ着"
        else generate_advice(class_label, reason)
    )

    return class_label, reason, advice