# interview-chat

## 開発用仮想環境

Python の仮想環境上で作業します。以下の手順で `.venv` を作成・利用してください。

1. 仮想環境の作成: `python3 -m venv .venv`
2. 仮想環境の有効化  
   - macOS / Linux: `source .venv/bin/activate`  
   - Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`
3. 仮想環境内で必要なライブラリをインストールしてください。 requirements.txt などが追加された場合は `pip install -r requirements.txt` を実行します。

作業終了後は `deactivate` で仮想環境を終了できます。

## Streamlit の起動

1. 上記の手順で仮想環境を有効化し、依存パッケージをインストールします。
2. リポジトリのルートで `streamlit run app/app.py` を実行します。
3. ブラウザで表示されるローカルホストの URL (通常は http://localhost:8501 ) にアクセスするとアプリを確認できます。
