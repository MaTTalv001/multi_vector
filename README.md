# Streamlit Application Template

## 概要

このレポジトリは、Streamlit アプリケーションの開発のためのテンプレートです。Docker 化された環境で、ホットリロードに対応しており、コードの変更がリアルタイムでアプリケーションに反映されます。

## 必要条件

- Docker
- Docker Compose

## プロジェクト構造

```

.
├── .gitignore
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .streamlit/ # Streamlit 設定ディレクトリ
│ ├── secrets.toml # 環境変数（gitignore 対象）
│ └── secrets.toml.template # 環境変数のテンプレート
├── src/ # アプリケーションのソースコード
│ ├── main.py # メインの Streamlit アプリケーション
│ ├── pages/ # 追加のページ
│ ├── config/ # 設定ファイル
│ └── utils/ # ユーティリティ関数
├── data/ # データファイル
│ ├── raw/ # 生データ
│ └── processed/ # 処理済みデータ
└── tests/ # テストコード

```

## セットアップと実行

### 環境変数の設定

1. `.streamlit/secrets.toml.template`をコピーして`.streamlit/secrets.toml`を作成します：

```bash
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
```

2. `.streamlit/secrets.toml`を編集して必要な環境変数を設定します。
   このファイルは`.gitignore`に含まれており、機密情報を安全に管理できます。

設定例（secrets.toml）：

```toml
# API Keys
openai_api_key = "your-api-key-here"

# Database Configuration
db_host = "localhost"
db_port = 5432
db_name = "mydatabase"

# Other Settings
debug = true
```

### アプリケーションの起動

```bash
# コンテナのビルドと起動
docker-compose up --build

# バックグラウンドで起動する場合
docker-compose up -d --build
```

アプリケーションは http://localhost:8501 でアクセスできます。

[以下、前の README と同様]

```

また、`.gitignore`に以下の行を追加することを推奨します：

```

# Streamlit secrets

.streamlit/secrets.toml

````

そして、`secrets.toml.template`には以下のような内容を記載することをお勧めします：

```toml
# API Keys
openai_api_key = "your-api-key-here"

# Application Settings
debug = true

# Add other configuration variables as needed
# database_url = "postgresql://user:password@localhost:5432/dbname"
# aws_access_key = "your-access-key"
# aws_secret_key = "your-secret-key"
````

このアプローチの利点：

1. Streamlit の公式推奨方法に従っている
2. 設定の一元管理が可能
3. テンプレートファイルにより必要な設定が明確
4. Docker 環境でも適切に動作する
5. セキュリティ的に安全（secrets.toml は gitignore される）
