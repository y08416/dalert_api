import sys
from model_loader import load_model
from explain import generate_explanation, generate_advice

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使い方: python app/cli_runner.py [画像ファイルパス]")
        sys.exit(1)

    image_path = sys.argv[1]

    print("モデル読み込み中...")
    model = load_model("model/model.h5")
    print("モデル読み込み完了！")

    label, reason = generate_explanation(model, image_path)
    advice = generate_advice(label, reason)

    print(f"この服は: {label}")
    print(f"理由: {reason}")
    print(f"アドバイス: {advice}")