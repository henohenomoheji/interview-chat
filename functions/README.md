# durable-excel-mini 概要

durable-excel-mini 配下のフォルダ構成と主要ファイルのコードを一覧化しています。仮想環境やキャッシュは省略しています。

## フォルダ構成

```
durable-excel-mini/
├─ function_app.py
├─ host.json
├─ local.settings.json
├─ requirements.txt
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ host.json
│  ├─ activities/
│  │  ├─ __init__.py
│  │  ├─ list_excel_files_activity.py
│  │  ├─ extract_sheets_activity.py
│  │  └─ combine_sheets_activity.py
│  └─ etl/
│     ├─ __init__.py
│     ├─ extract_excel.py
│     └─ combine.py
└─ .venv/, __pycache__/ ...（省略）
```

## コード一覧

### function_app.py

```python
import azure.functions as func
import azure.durable_functions as df

# activities の実装は別モジュールのまま使う（名前衝突を避けるため alias）
from app.activities import (
    list_excel_files_activity as a_list,
    extract_sheets_activity as a_extract,
    combine_sheets_activity as a_combine,
)

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# HTTPでOrchestrator起動
@app.route(route="orchestrators/{name}")
@app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client):
    name = req.route_params.get("name")
    instance_id = await client.start_new(name)
    return client.create_check_status_response(req, instance_id)

# Orchestrator：①抽出 → ②結合
@app.orchestration_trigger(context_name="context")
def excel_extract_and_combine_orchestrator(context: df.DurableOrchestrationContext):
    # Activity 名は「デコレータを付けた関数名」になるので、
    # call_activity の文字列もそれに合わせる
    files = yield context.call_activity("list_excel_files_activity", None)

    tasks = [context.call_activity("extract_sheets_activity", f) for f in files]
    extracted_per_file = yield context.task_all(tasks)

    combined = yield context.call_activity("combine_sheets_activity", extracted_per_file)

    preview = combined[:20]
    return {
        "files": len(files),
        "rows": len(combined),
        "preview": preview
    }

# --- Activity（デコレータで登録する） ---
@app.activity_trigger(input_name="dummy")
def list_excel_files_activity(dummy):
    return a_list.main(dummy)

@app.activity_trigger(input_name="file_info")
def extract_sheets_activity(file_info: dict):
    return a_extract.main(file_info)

@app.activity_trigger(input_name="extracted_per_file")
def combine_sheets_activity(extracted_per_file):
    return a_combine.main(extracted_per_file)
```

### app/config.py

```python
import os
from pathlib import Path


def get_excel_dir() -> str:
    d = os.getenv("LOCAL_EXCEL_DIR", "./data/excel")
    return str(Path(d).resolve())
```

### app/activities/list_excel_files_activity.py

```python
from pathlib import Path
from app.config import get_excel_dir

def main(_):
    directory = Path(get_excel_dir())
    print(f"directory:{directory}")
    if not directory.exists():
        print("directoryが存在しない")
        return []

    files = []
    for fp in directory.glob("*.xlsx"):
        files.append({"path": str(fp), "name": fp.name})

    print(f"files:{files}")
    return files
```

### app/activities/extract_sheets_activity.py

```python
from pathlib import Path
from app.etl.extract_excel import extract_all_sheets


def main(file_info: dict):
    fp = Path(file_info["path"])
    excel_bytes = fp.read_bytes()
    return extract_all_sheets(excel_bytes, source_name=file_info["name"])
```

### app/activities/combine_sheets_activity.py

```python
from app.etl.combine import combine_extracted


def main(extracted_per_file):
    return combine_extracted(extracted_per_file)
```

### app/etl/extract_excel.py

```python
import io
import pandas as pd


def extract_all_sheets(excel_bytes: bytes, source_name: str) -> list[dict]:
    """
    戻り値:
      [
        {"file": "a.xlsx", "sheet": "Sheet1", "records": [ {...}, {...} ]},
        ...
      ]
    """
    buf = io.BytesIO(excel_bytes)
    xl = pd.ExcelFile(buf, engine="openpyxl")

    out: list[dict] = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet_name=sheet)

        # まっさらな空行だけ除去（必要なら調整）
        df = df.dropna(how="all")

        # 由来情報を追加
        df["__source_file"] = source_name
        df["__source_sheet"] = sheet

        out.append({
            "file": source_name,
            "sheet": sheet,
            "records": df.to_dict(orient="records")
        })
    return out
```

### app/etl/combine.py

```python
import pandas as pd


def combine_extracted(extracted_per_file: list[list[dict]]) -> list[dict]:
    """
    extracted_per_file:
      [
        [ {"file","sheet","records"}, ... ],  # file1
        [ ... ],                               # file2
      ]

    結合方針（最小）:
      - 全シート・全ファイルの records を全部1つにまとめる
      - 返却は list[dict]
    """
    # None/空配列の場合はそのまま返す
    if not extracted_per_file:
        return []

    frames = []
    for per_file in extracted_per_file:
        if not per_file:
            continue
        for block in per_file:
            if block.get("records"):
                frames.append(pd.DataFrame(block["records"]))

    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)

    # まずは軽く整形（列名strip、NaN→空文字）
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("")

    return df.to_dict(orient="records")
```

### host.json

```json
{
  "version": "2.0",
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  }
}
```

### app/host.json

```json
{
  "version": "2.0",
  "isDefaultHostConfig": true,
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  }
}
```

### local.settings.json

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "LOCAL_EXCEL_DIR": "/workspaces/test-analysis/data/excel"
  }
}
```

### requirements.txt

```
azure-functions
azure-functions-durable
pandas
openpyxl
```

起動

```
curl -X POST   http://localhost:7071/api/orchestrators/excel_extract_and_combine_orchestrator   -H "Content-Type: application/json"   -d '{
    "path": "/workspaces/test-analysis/data/excel/sample.xlsx"
  }'
```

