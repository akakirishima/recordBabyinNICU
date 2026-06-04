import datetime
import matplotlib.pyplot as plt
import csv

# --- 設定 ---
INPUT_CSV = "momentum_results.csv"  # 読み込むCSVファイル名
Y_MAX = 200.0                         # 縦軸(運動量)の最大固定値（論文比較用。必要に応じて調整）

# 1. CSVからデータを読み込んで1時間ごとにグループ化
hourly_data = {}

print(f"{INPUT_CSV} を読み込み中...")

with open(INPUT_CSV, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # ヘッダー行をスキップ
    
    for i, row in enumerate(reader):
        # 10行に1行（約0.3秒ごと）に間引いて読み込み、処理と描画を軽くする
        if i % 10 == 0:
            try:
                # 文字列をdatetime型とfloat型に変換
                dt = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
                val = float(row[1])
                
                # 「日付_時間」を辞書のキーにする (例: "20260603_15")
                hour_key = dt.strftime("%Y%m%d_%H")
                
                if hour_key not in hourly_data:
                    hourly_data[hour_key] = {"times": [], "values": []}
                
                hourly_data[hour_key]["times"].append(dt)
                hourly_data[hour_key]["values"].append(val)
                
            except ValueError:
                continue

print("読み込み完了。1時間ごとに分割してグラフを表示・保存します。")

# 2. グループごとに個別グラフを出力・表示
for hour_key, data in hourly_data.items():
    print(f"表示中: {hour_key}時台 （ウィンドウを閉じると次の時間帯に進みます）")
    
    plt.figure(figsize=(12, 4))
    plt.plot(data["times"], data["values"], color="crimson", linewidth=0.7, label="Momentum")
    
    # フォーマット調整
    plt.gcf().autofmt_xdate()  # 横軸の日時表示（傾き）を綺麗にする
    plt.xlabel("Absolute Time (YYYY-MM-DD HH:MM)")
    plt.ylabel("Momentum (kg·m/s)")
    plt.title(f"Momentum Plot - {hour_key}:00")
    
    # 【論理的改善】論文用に全てのグラフの縦軸を統一
    plt.ylim(0, Y_MAX) 
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # 2-1. 画像として自動保存 (例: momentum_plot_20260603_15.png)
    plt.savefig(f"momentum_plot_{hour_key}.png", dpi=150)
    
    # 2-2. 画面にグラフを表示（×で閉じられるまでプログラムを一時停止する）
    plt.show()

print("すべての時間帯のグラフ表示・保存が完了しました。")
