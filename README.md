# Dalert API - ダル着 or おしゃれ着判定AI

Dalertは、服装画像を入力として  
「ダル着（部屋着）」か「おしゃれ着」かを判定し、  
さらに理由とアドバイスを返す **FastAPIベースのバックエンドAPI** です。

---

## 概要

| 項目 | 内容 |
|------|------|
| 名称 | Dalert API |
| 目的 | ユーザーが自分の服装をAIで分析し、改善ヒントを得るための判定API |
| 判定内容 | 「ダル着」 or 「おしゃれ着」 |
| 出力 | 判定結果 + 理由 + アドバイス（自然文） |
| 開発環境 | Python / FastAPI / TensorFlow / OpenAI API |
| モデル | MobileNetV2（転移学習） |
| デプロイ環境 | Render（FastAPIアプリとして稼働） |

---

## 公開APIエンドポイント

本番APIは Render 上で稼働しています。

- ベースURL  
  `https://dalert-api.onrender.com`

- Swagger UI（APIドキュメント）  
  `https://dalert-api.onrender.com/docs`

### 利用例

POST `/predict`

```bash
curl -X POST "https://dalert-api.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg"
```

レスポンス例:

```json
{
  "label": "おしゃれ着",
  "reason": "上半身の明るいトーンとバランスの取れた配色が好印象です。",
  "advice": "差し色を足すとより洗練された印象になります。"
}
```

---

## エンドポイント仕様

### POST /predict

画像を受け取り、服装の判定・理由・アドバイスを返します。

| フィールド | 型 | 説明 |
|-------------|----|------|
| `file` | `multipart/form-data` | JPEG/PNG 形式の服装画像 |

**Status Codes**
- `200 OK` : 正常に推論完了
- `400 Bad Request` : 画像ファイルなし
- `500 Internal Server Error` : 推論またはAPIエラー

---

## 環境構築（ローカル開発用）

### 1. 仮想環境セットアップ

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. OpenAI APIキー設定

`.env` をルートに作成して以下を記入。

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. モデル学習（初回のみ）

```bash
python scripts/train.py
```

学習完了後、`model/model.h5` が生成されます。

### 4. API起動（ローカル）

```bash
uvicorn app.main:app --reload
```

- ローカルURL: http://localhost:8000  
- Swagger UI: http://localhost:8000/docs  

---

## ディレクトリ構成

```text
DALERT_PROTOTYPE/
├── app/                # API・推論関連コード
│   ├── main.py         # FastAPIエントリーポイント
│   ├── explain.py      # 画像解析と理由生成
│   ├── model_loader.py # モデル読み込み・推論処理
│   └── cli_runner.py   # CLI実行テストスクリプト
│
├── scripts/
│   └── train.py        # 転移学習によるモデル学習スクリプト
│
├── model/              # 学習済みモデル保存先
│   └── model.h5
│
├── data/               # 学習データセット
│   ├── dalugi/         # ダル着画像
│   └── osyaregi/       # おしゃれ着画像
│
├── tmp_uploads/        # 一時アップロードフォルダ
│
├── .env                # OpenAI APIキーなどの環境変数
├── .gitignore
├── requirements.txt     # 依存パッケージ一覧
└── README.md            # プロジェクト説明
```

---

## 処理フロー

1. クライアントから画像を `/predict` にPOST  
2. FastAPIが画像を一時保存  
3. TensorFlowモデル（MobileNetV2）が推論を実行  
4. Grad-CAMで注目部位を推定  
5. HSV色空間で彩度・明度を解析  
6. OpenAI GPT-3.5で理由文・アドバイスを生成  
7. JSONでレスポンスを返却  

---

## CLIテスト（ローカル確認）

```bash
python app/cli_runner.py sample.jpg
```

出力例:

```
モデル読み込み中...
モデル読み込み完了！
この服は: おしゃれ着
理由: 上半身の明るい色使いが印象的だったため
アドバイス: 落ち着いた色を加えるとよりバランスが良くなります
```

---

## Author

尾﨑 陽介（Yosuke Ozaki）  
Ritsumeikan University 
Backend Engineer / Media & AI Researcher

---

## License

MIT License

---