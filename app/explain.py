import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def generate_explanation(model, img_path, last_conv_layer_name="Conv_1"):
    img_size = (224, 224)

    # 画像読み込み
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Grad-CAMの準備
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

    # ヒートマップ生成（.numpy()は使わない）
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam + 1e-8)  # 安定化
    cam = cv2.resize(np.array(cam), img_size)

    # 注目領域の重心を計算
    heatmap_thresh = np.where(cam > 0.5, 1, 0).astype(np.uint8)
    moments = cv2.moments(heatmap_thresh)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = img_size[0] // 2, img_size[1] // 2  # fallback

    # 位置ベースの判定
    if cy < img_size[1] * 0.4:
        position = "上半身"
    elif cy > img_size[1] * 0.6:
        position = "下半身"
    else:
        position = "全体"

    # カラー特徴の取得
    img_cv = cv2.imread(img_path)
    img_resized = cv2.resize(img_cv, img_size)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hue_mean = hsv[..., 0].mean()
    saturation_mean = hsv[..., 1].mean()

    if saturation_mean > 100:
        color_desc = "鮮やかな色味"
    else:
        color_desc = "落ち着いた色味"

    # クラスに応じた文章生成
    class_label = "おしゃれ着" if int(pred_index) == 1 else "ダル着"

    reason = f"{position}の{color_desc}に注目し、{class_label}と判断しました。"
    return class_label, reason