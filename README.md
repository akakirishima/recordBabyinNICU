# recordBabyinNICU

Intel RealSense L515 を使って、NICU 環境で Depth、IR、RGB の各ストリームを記録するための Python スクリプト群です。  
公開リポジトリとしての主役は [`H_rgb_ir_depth.py`](./H_rgb_ir_depth.py) で、Depth、IR、RGB を同時に保存します。  
帯域や欠損リスクを抑えたい場合は、安定優先プロファイルの [`stable_rgb_ir_depth.py`](./stable_rgb_ir_depth.py) を使えます。

このリポジトリは録画コードの公開を目的としており、患者データや録画済みデータセットを公開するものではありません。

## Overview

[`H_rgb_ir_depth.py`](./H_rgb_ir_depth.py) は、Intel RealSense L515 から取得した 3 つのストリームを連続録画します。

| Stream | Format | Default behavior in `H_rgb_ir_depth.py` |
| --- | --- | --- |
| Depth | HDF5 (`uint16`) | `1024x768 @ 30 fps` を全フレーム保存 |
| IR | MP4 | `1024x768 @ 30 fps` を保存 |
| RGB | MP4 | `1920x1080 @ 30 fps` を保存 |
| Session info | TXT | 開始時に機材情報と入力値を保存 |

録画ファイルは一定時間ごとに分割され、日付と時刻に基づくフォルダ構成で保存されます。

## Requirements

- Windows 環境を前提にしたスクリプトです
- Intel RealSense L515 実機
- Python 3 系
- `pyrealsense2`
- `opencv-python`
- `h5py`
- `numpy`

Python パッケージの固定バージョンは [`requirements.txt`](./requirements.txt) を参照してください。

RealSense SDK / `pyrealsense2` の導入やデバイス設定は、環境によって追加セットアップが必要な場合があります。

- Intel RealSense SDK: [librealsense](https://github.com/IntelRealSense/librealsense)

## Installation

1. リポジトリを取得します。

```bash
git clone https://github.com/akakirishima/recordBabyinNICU.git
cd recordBabyinNICU
```

2. 仮想環境を作成します。

```bash
python -m venv .venv
```

3. 仮想環境を有効化します。

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. 依存パッケージをインストールします。

```bash
pip install -r requirements.txt
```

## Usage

まず [`H_rgb_ir_depth.py`](./H_rgb_ir_depth.py) 内の保存先設定を、自分の環境に合わせて確認してください。

```python
ROOT_PATH        = r"D:/Dev/Data"
DEPTH_W, DEPTH_H = 1024, 768
IR_W,    IR_H    = 1024, 768
RGB_W,   RGB_H   = 1920, 1080
FPS              = 30
FILE_PERIOD_MIN  = 1
VISUALIZE        = False
```

実行:

```bash
python H_rgb_ir_depth.py
```

起動時に以下を入力します。

- `baby ID`
- `PC name`

主な挙動:

- L515 が未接続の場合は終了します
- 録画開始前にウォームアップを行います
- `FILE_PERIOD_MIN` ごとに出力ファイルを分割します
- `VISUALIZE = True` の場合はプレビューウィンドウを表示します
- プレビュー中に `q` を押すと停止します
- `Ctrl+C` でも停止できます

欠損リスクを抑えたい場合は、安定優先版も使えます。

```bash
python stable_rgb_ir_depth.py
```

`stable_rgb_ir_depth.py` のデフォルトは `RGB 1280x720 @ 30 fps`、`IR 1024x768 @ 15 fps`、`Depth 1024x768 @ 5 fps save target` です。  
backlog 時は `RGB > IR > Depth` の優先度で、Depth、次に IR を先に削る設計です。

## Output Format

[`H_rgb_ir_depth.py`](./H_rgb_ir_depth.py) の出力構造は以下です。

```text
ROOT_PATH/
  babyID/
    YYYYMMDD/
      YYYYMMDD_HHMMSS_info.txt
      Depth/
        HH/
          YYYYMMDD_HHMMSS.h5
      IR/
        HH/
          YYYYMMDD_HHMMSS.mp4
      RGB/
        HH/
          YYYYMMDD_HHMMSS.mp4
```

HDF5 には主に次の情報が入ります。

- `depth`: 各フレームの深度データ
- `ts`: RealSense 由来のタイムスタンプ
- attributes: `depth_scale`, `width`, `height`, `fps`, `serial`

`*_info.txt` には次のような実行情報が保存されます。

- `babyID`
- `pcname`
- RealSense serial number
- firmware version
- 各ストリームの解像度と FPS
- 開始時刻

## Scripts

このリポジトリには用途の異なる派生スクリプトも含まれます。

- [`H_rgb_ir_depth.py`](./H_rgb_ir_depth.py): 推奨スクリプト。Depth、IR、RGB をまとめて記録
- [`stable_rgb_ir_depth.py`](./stable_rgb_ir_depth.py): 安定優先版。`RGB > IR > Depth` の優先度で backlog 時に Depth、次に IR を先に削る
- [`H_rgb_ir.py`](./H_rgb_ir.py): IR と RGB のみを記録
- [`H_depth_ir_autosplit.py`](./H_depth_ir_autosplit.py): Depth と IR を中心に扱う派生版
- [`lossless_RGB_IR_depth.py`](./lossless_RGB_IR_depth.py): 保存形式や FPS が異なる実験的バリエーション
- [`preview.py`](./preview.py): 接続確認とプレビュー用

README の説明対象は主に `H_rgb_ir_depth.py` です。各スクリプトは保存形式、FPS、解像度、プレビュー設定が異なるため、使用前に先頭の設定値を確認してください。

## Limitations

- コマンドライン引数ではなく、スクリプト内定数で設定する設計です
- 保存先 `ROOT_PATH` はコード内で固定されています
- Windows パスを前提にした初期設定が含まれます
- L515 実機がない環境では動作確認できません
- 自動テストや再生サンプルは現状含まれていません

## Privacy / Data Handling

このリポジトリは NICU 文脈の録画コードを含むため、データの取り扱いに注意してください。

- 実際の患者データ、映像、深度データ、識別子を含むファイルは公開リポジトリに含めないでください
- `baby ID` や `PC name` は保存ファイルに書き込まれるため、実運用では識別ポリシーを明確にしてください
- サンプルデータを公開する場合は、匿名化または十分な非識別化を行ったものだけを使用してください
- セキュリティやプライバシーに関わる内容を public issue に投稿しないでください

参考:

- GitHub Docs: [About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
- GitHub Docs: [Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- HHS: [Guidance Regarding Methods for De-identification of Protected Health Information](https://www.hhs.gov/sites/default/files/ocr/privacy/hipaa/understanding/coveredentities/De-identification/hhs_deid_guidance.pdf)

## Development Notes

依存関係を更新した場合:

```bash
pip install <package>
pip freeze > requirements.txt
```

録画データをリポジトリ配下に置く運用は推奨しません。やむを得ず置く場合は、[`.gitignore`](./.gitignore) の除外設定を必ず確認してください。

## License

このリポジトリは MIT License で公開します。詳細は [`LICENSE`](./LICENSE) を参照してください。
