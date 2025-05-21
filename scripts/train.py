import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 1. パラメータ設定
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DALUGI_DIR = os.path.join(DATA_DIR, 'dalugi')
OSYAREGI_DIR = os.path.join(DATA_DIR, 'osyaregi')
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.h5')

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

# 2. データ読み込み
def load_images():
    data = []
    labels = []
    
    for category, label in [('dalugi', 0), ('osyaregi', 1)]:
        folder = os.path.join(DATA_DIR, category)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                image = load_img(file_path, target_size=IMG_SIZE)
                image = img_to_array(image)
                data.append(image)
                labels.append(label)
            except Exception as e:
                print(f"エラー: {file_path}, {e}")
    
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels

# 3. データ準備
print("画像読み込み中...")
X, y = load_images()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. モデル作成（転移学習）
print("モデル構築中...")
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# 5. コンパイルと学習
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

print("学習開始！！")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# 6. モデル保存
print(f"モデル保存中... -> {MODEL_OUTPUT_PATH}")
model.save(MODEL_OUTPUT_PATH)

print("✅ 全部完了！！！")