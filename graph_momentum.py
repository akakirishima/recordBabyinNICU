import datetime
import matplotlib.pyplot as plt
import csv

# --- 設定 ---
INPUT_CSV = "momentum_results.csv"  # 読み込むCSVファイル名
Y_MAX = 200.0                         # 縦軸(運動量)の最大固定値

# 1. CSVからデータを読み込んでグループ化
hourly_data = {}
all_times = []
all_values = []

print(f"{INPUT_CSV} を読み込み中...")

with open(INPUT_CSV, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # ヘッダー行をスキップ
    
    for i, row in enumerate(reader):
        # 10行に1行（約0.3秒ごと）に間引いて読み込み、処理と描画を軽くする
        if i % 10 == 0:
            try:
                dt = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
                val = float(row[1])
                
                # まとめグラフ用の全データリストに格納
                all_times.append(dt)
                all_values.append(val)
                
                # 1時間ごとのグループ分け
                hour_key = dt.strftime("%Y%m%d_%H")
                if hour_key not in hourly_data:
                    hourly_data[hour_key] = {"times": [], "values": []}
                
                hourly_data[hour_key]["times"].append(dt)
                hourly_data[hour_key]["values"].append(val)
                
            except ValueError:
                continue

print("読み込み完了。グラフを一括生成しています...")

# 2. 1時間ごとの個別グラフを裏で作成・保存
for hour_key, data in hourly_data.items():
    plt.figure(num=f"Graph_{hour_key}", figsize=(12, 4))
    plt.plot(data["times"], data["values"], color="crimson", linewidth=0.7, label="Momentum")
    
    plt.gcf().autofmt_xdate()
    plt.xlabel("Absolute Time (YYYY-MM-DD HH:MM)")
    plt.ylabel("Momentum (kg·m/s)")
    plt.title(f"Momentum Plot - {hour_key}:00")
    
    plt.ylim(0, Y_MAX) 
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"momentum_plot_{hour_key}.png", dpi=150)
    print(f"画像保存完了: momentum_plot_{hour_key}.png")

# 3. 【追加】12時間分をまとめた「全通しグラフ」の作成・保存
if all_times:
    print("12時間まとめグラフを生成中...")
    plt.figure(num="Graph_12Hour_Summary", figsize=(16, 5))
    # 全体を俯瞰するため、色はダークネイビー、線は細めでスタイリッシュに描画
    plt.plot(all_times, all_values, color="midnightblue", linewidth=0.4, label="12-Hour Continuous Momentum")
    
    plt.gcf().autofmt_xdate()
    plt.xlabel("Absolute Time (HH:MM)")
    plt.ylabel("Momentum (kg·m/s)")
    plt.title("12-Hour Continuous Momentum Summary Plot")
    
    plt.ylim(0, Y_MAX) 
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # まとめ用画像として保存
    plt.savefig("momentum_plot_12hours_summary.png", dpi=150)
    print("画像保存完了: momentum_plot_12hours_summary.png")

print("\n--- 全解析が完了しました。すべてのグラフ（個別＋まとめ）を同時に表示します ---")
# これまでの個別12枚＋まとめ1枚の計13枚のウィンドウを一斉に開く
plt.show()
