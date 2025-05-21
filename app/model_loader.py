import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# モデルのパスを指定
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.h5')

# モデルロード
print("モデル読み込み中...")
model = load_model(MODEL_PATH)
print("モデル読み込み完了！")

# 画像判定用関数
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # 正規化

    prediction = model.predict(img)[0][0]

    if prediction < 0.5:
        return "ダル着"
    else:
        return "外着（おしゃれ着）"