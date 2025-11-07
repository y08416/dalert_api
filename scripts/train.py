import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# パラメータ設定
# └ データセットのパスやハイパーパラメータを定義する。
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DALUGI_DIR = os.path.join(DATA_DIR, 'dalugi')      # 「ダル着」画像フォルダ
OSYAREGI_DIR = os.path.join(DATA_DIR, 'osyaregi')  # 「おしゃれ着」画像フォルダ
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.h5')

IMG_SIZE = (224, 224)  # MobileNetV2の標準入力サイズ
BATCH_SIZE = 8
EPOCHS = 10


# データ読み込み関数
# └ 各フォルダから画像を読み込み、数値化してラベル（0: ダル着, 1: おしゃれ着）を付与する。
def load_images():
    data = []
    labels = []
    
    for category, label in [('dalugi', 0), ('osyaregi', 1)]:
        folder = os.path.join(DATA_DIR, category)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                # 画像を224×224にリサイズして読み込み
                image = load_img(file_path, target_size=IMG_SIZE)
                # NumPy配列に変換
                image = img_to_array(image)
                data.append(image)
                labels.append(label)
            except Exception as e:
                # 破損画像などはスキップ
                print(f"エラー: {file_path}, {e}")
    
    # NumPy配列化と正規化（0〜1）
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels


# データ準備
# └ 全データを読み込み、学習用と検証用に分割する。
print("画像読み込み中...")
X, y = load_images()

# 学習データ80%、検証データ20%で分割
# └ ランダムシード(random_state=42)を固定して再現性を確保
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# モデル構築（転移学習）
# └ MobileNetV2をベースに、上に独自の分類層を追加。
#    転移学習を使うことで、少ないデータでも高精度な特徴抽出が可能。
print("モデル構築中...")
base_model = MobileNetV2(
    weights="imagenet",      # 既存の学習済み重みを利用（汎用的な特徴を転用）
    include_top=False,       # 既存の全結合層は使わず、独自の分類層を上に追加
    input_shape=(224, 224, 3)
)

# 出力層の構成
# └ GlobalAveragePooling2D: 畳み込み出力を平均化して特徴ベクトル化
# └ Dense(128, relu): 中間層で学習
# └ Dense(1, sigmoid): 出力が0〜1の範囲（2値分類）
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 転移学習では、まずベースモデルの重みを固定
# └ これによりImageNetで学習済みの汎用特徴（エッジ・形・色など）を保持する。
for layer in base_model.layers:
    layer.trainable = False


# モデルのコンパイル
# └ 損失関数: binary_crossentropy（2値分類）
# └ 最適化: Adam（学習の安定性が高い）
# └ 評価指標: accuracy（正解率）
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])


# モデルの学習
# └ 画像配列を直接fit()に渡して学習。
#    データ数が少ない場合はData Augmentationも有効（ImageDataGeneratorで拡張可）。
print("学習開始！！")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# モデル保存
# └ 学習済み重みと構造をまとめて保存。
#    保存後はFastAPI側で `load_model()` で読み込める。
print(f"モデル保存中... -> {MODEL_OUTPUT_PATH}")
model.save(MODEL_OUTPUT_PATH)

print("✅ 全部完了！！！")