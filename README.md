# recordBabyinNICU

本プロジェクトは、Intel RealSense L515 デバイスから取得した深度 (Depth)、赤外線 (IR)、カラー (RGB) の各ストリームを同期録画し、深度データを HDF5、IR/RGB データを MP4 フォーマットで保存するものです。

---

## 📦 セットアップ

1. **リポジトリをクローン**

   ```bash
   git clone <リポジトリのURL>
   cd recordBabyinNICU
   ```
2. **仮想環境を作成**

   ```bash
   python -m venv .venv
   ```
3. **仮想環境を有効化**

   * **PowerShell (Windows)**

     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **CMD (Windows)**

     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   * **bash/zsh (macOS/Linux)**

     ```bash
     source .venv/bin/activate
     ```
4. **依存パッケージをインストール**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 使用方法

仮想環境を有効化した状態で、以下を実行します。

```bash
python H_rgb_ir_depth.py
```

実行後に表示されるプロンプトで **baby ID** と **PC名** を入力すると、録画が開始されます。

---

## 🛠 開発・管理

* **依存関係を固定**

  ```bash
  pip freeze > requirements.txt
  ```
* **ライブラリ追加**

  ```bash
  pip install <パッケージ名>
  pip freeze > requirements.txt
  ```

---

## 🚫 .gitignore

以下の設定を `.gitignore` に追加し、ローカル環境固有のファイルをコミット対象から除外してください。

```gitignore
# 仮想環境
.venv/
venv/

# Python キャッシュ・バイトコード
__pycache__/
*.py[cod]
*$py.class

# エディタ / IDE
.vscode/
.idea/
```

