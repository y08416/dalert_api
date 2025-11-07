import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from openai import OpenAI
from PIL import Image
import os
from dotenv import load_dotenv
import pillow_heif

# HEIF/HEIC形式の画像を開けるように設定（iPhone写真対応）
pillow_heif.register_heif_opener()

# OpenAI APIキーを.envファイルから読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# アドバイス生成
def generate_advice(label: str, reason: str) -> str:
    """
    モデルの判定結果と理由をもとに、
    GPTを使って「さらにおしゃれに見せるにはどうすればいいか」を1文で生成する。
    """
    prompt = f"""
    この服装は「{label}」と判断されました。理由は「{reason}」です。
    さらにおしゃれにするにはどうすればよいか、1文で具体的なアドバイスをください。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  # 出力の多様性を少し確保
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API error (advice):", e)
        return "アドバイス生成に失敗しました。"


# 理由文生成
def generate_reason(label: str, position: str, color_desc: str) -> str:
    """
    Grad-CAMと色情報をもとに、
    GPTで「どこを見て、どんな特徴からその判定になったか」を自然な日本語で生成する。
    """
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
        print("OpenAI API error (reason):", e)
        return f"{position}の{color_desc}に注目し、{label}と判断しました。"


# 説明生成（Grad-CAM + 色特徴 + GPT）
def generate_explanation(model, img_path, last_conv_layer_name="Conv_1"):
    """
    入力画像をモデルで解析し、Grad-CAMで注目領域を可視化。
    その領域の色情報（HSV）を解析して「おしゃれ or ダル着」を判定。
    最後にGPTで自然な説明文とアドバイスを生成する。
    """

    # モデル入力サイズ（MobileNetV2などと同じ224×224）
    img_size = (224, 224)

    # 画像前処理
    # └ モデルに入力できる形式に整える。学習時と同じ正規化を適用する。
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # 0〜1にスケーリング

    # Grad-CAMモデル構築
    # └ 最終畳み込み層と出力層の両方を取得できるようにする。
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 勾配の計算
    # └ 特徴マップがどの程度最終出力に寄与しているかを算出する。
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # 予測確率が最も高いクラスに対する勾配を取得
        class_channel = predictions[:, tf.argmax(predictions[0])]

    # Conv層出力に対する勾配を計算（特徴量ごとの影響度）
    grads = tape.gradient(class_channel, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    # 各チャネルの平均勾配を求めて重みとする（Grad-CAMの核心）
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Grad-CAMマップの生成
    # └ 各特徴マップを重み付きで合成し、モデルの注目領域を可視化する。
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]
    cam = np.maximum(cam, 0)       # 負の値を0に（ReLU）
    cam = cam / np.max(cam + 1e-8) # 正規化（0〜1）
    cam = cv2.resize(cam, img_size)

    # 注目位置の推定
    # └ Grad-CAMの重心を求めて「上半身・下半身・全体」を分類する。
    heatmap_thresh = np.where(cam > 0.5, 1, 0).astype(np.uint8)
    moments = cv2.moments(heatmap_thresh)
    if moments["m00"] != 0:
        cy = int(moments["m01"] / moments["m00"])  # y方向の重心
    else:
        cy = img_size[1] // 2                      # 見つからなければ中央扱い

    if cy < img_size[1] * 0.4:
        position = "上半身"
    elif cy > img_size[1] * 0.6:
        position = "下半身"
    else:
        position = "全体"

    # 色特徴解析（HSV空間）
    # └ HSVは人間の感覚に近い色空間で、彩度や明度を独立に扱える。
    pil_img = Image.open(img_path).convert("RGB")
    img_np = np.array(pil_img.resize(img_size))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Grad-CAMで強調された領域をマスクし、その部分の色特徴だけを抽出
    mask = (cam > 0.5).astype(np.uint8)
    masked_hsv = hsv * np.expand_dims(mask, axis=2)

    # マスク内の各チャンネル（色相・彩度・明度）を取得
    hue_vals = masked_hsv[..., 0][mask == 1]
    sat_vals = masked_hsv[..., 1][mask == 1]
    val_vals = masked_hsv[..., 2][mask == 1]

    # 領域全体の平均値を算出し、色の傾向を定量化
    hue_mean = hue_vals.mean() if hue_vals.size > 0 else 0
    sat_mean = sat_vals.mean() if sat_vals.size > 0 else 0
    val_mean = val_vals.mean() if val_vals.size > 0 else 0

    # 色味の記述
    # └ 平均HSV値から感性的に理解しやすい表現に変換する。
    if sat_mean > 100:  # 彩度が高ければ鮮やかな印象
        if hue_mean < 30 or hue_mean > 150:
            color_desc = "暖色系で鮮やかな色味"
        else:
            color_desc = "寒色系で鮮やかな色味"
    else:
        color_desc = "落ち着いた色味"

    # 判定ロジック
    # └ 彩度・明度・注目位置を組み合わせて「おしゃれ／ダル着」を分類。
    #    ・彩度(sat)：色の鮮やかさ。低いと地味でダル着っぽく見える。
    #    ・明度(val)：明るさ。低いと暗い印象になりやすい。
    #    ・位置(position)：明暗のバランスで印象が変わるため補助要素に使う。
    if sat_mean < 100 and val_mean < 140:
        # 全体が暗くくすんでいる服装 → ダル着判定
        if position == "全体":
            class_label = "ダル着"
        else:
            class_label = "おしゃれ着"
    elif sat_mean < 80 and val_mean < 160 and position != "下半身":
        # 明度が低めで彩度が抑えられた上半身中心の服 → ダル着
        class_label = "ダル着"
    else:
        # それ以外（明るく鮮やか、または上下のバランス良好） → おしゃれ着
        class_label = "おしゃれ着"

    # 理由とアドバイス生成
    # └ GPTを使って自然な日本語文に整える。
    reason = generate_reason(class_label, position, color_desc)
    advice = (
        "特に改善点はありません。今のままで十分おしゃれです！"
        if class_label == "おしゃれ着"
        else generate_advice(class_label, reason)
    )

    # 判定結果・理由文・アドバイスを返す
    return class_label, reason, advice