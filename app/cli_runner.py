import sys
from model_loader import load_model
from explain import generate_explanation, generate_advice

# エントリーポイント（直接実行されたときだけ動く）
if __name__ == "__main__":
    # 引数チェック（画像パスが指定されてるか）
    if len(sys.argv) != 2:
        print("使い方: python app/cli_runner.py [画像ファイルパス]")
        sys.exit(1)

    # コマンドライン引数から画像パスを取得
    image_path = sys.argv[1]

    print("モデル読み込み中...")
    # モデルをロード（FastAPIと同じやつ）
    model = load_model("model/model.h5")
    print("モデル読み込み完了！")

    # モデル推論 + 理由文生成
    label, reason = generate_explanation(model, image_path)

    # ラベルと理由からアドバイスを生成
    advice = generate_advice(label, reason)

    # 結果をコンソール出力
    print(f"この服は: {label}")
    print(f"理由: {reason}")
    print(f"アドバイス: {advice}")