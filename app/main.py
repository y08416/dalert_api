import sys
from model_loader import load_model
from explain import generate_explanation

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使い方: python app/main.py [画像ファイルパス]")
        sys.exit(1)

    image_path = sys.argv[1]

    print("モデル読み込み中...")
    model = load_model()
    print("モデル読み込み完了！")

    label, reason = generate_explanation(model, image_path)
    print(f"この服は: {label}")
    print(f"理由: {reason}")