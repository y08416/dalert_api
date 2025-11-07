import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# モデルパスの指定
# └ __file__ で現在のファイル位置を取得し、相対パスで model/model.h5 を指す。
#    これにより、実行ディレクトリがどこでも安定してモデルを読み込める。
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.h5')

# モデルのロード
# └ Kerasのload_modelを使って学習済みモデル（CNNなど）を読み込む。
#    ここで一度だけロードして、以降の推論で再利用する。
print("モデル読み込み中...")
model = load_model(MODEL_PATH)
print("モデル読み込み完了！")

# 画像判定関数
# └ 与えられた画像を前処理してモデルに入力し、
#    出力値に基づいて「ダル着」か「おしゃれ着」を判定する。
def predict_image(image_path):
    # 画像読み込みとリサイズ
    # └ モデル学習時と同じサイズ（224×224）に揃える。
    img = load_img(image_path, target_size=(224, 224))
    
    # NumPy配列に変換（float型にして後処理しやすくする）
    img = img_to_array(img)
    
    # バッチ次元を追加（shapeを(1, 224, 224, 3)に）
    # └ Kerasモデルは常に「複数画像」を前提にしているため。
    img = np.expand_dims(img, axis=0)
    
    # 正規化
    # └ 学習時と同じく、画素値を0〜1の範囲にスケーリング。
    #    こうすることでモデルの重みが安定して動作する。
    img = img / 255.0

    # 推論実行
    # └ モデル出力は通常 [確率] の形で返される（例: 0〜1の範囲）。
    prediction = model.predict(img)[0][0]

    # 判定
    # └ 出力が0.5未満なら「ダル着」、それ以上なら「おしゃれ着」と判断。
    #    閾値0.5は2値分類の標準的な境界。
    if prediction < 0.5:
        return "ダル着"
    else:
        return "外着（おしゃれ着）"