import os
import glob
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import re

# --- 設定 ---
BASE_DIR = "."                 # test.pyがあるディレクトリ（カレントフォルダ）
OUTPUT_CSV = "momentum_results.csv"
OBJECT_MASS = 1.0              # 物体の質量 (kg)
FPS = 30                       # 動画のフレームレート

# カメラの内部パラメータ (L515の標準値例)
FX, FY, CX, CY = 460.0, 460.0, 320.0, 240.0

# CSVファイルの初期化
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Momentum(kg·m/s)"])

def pixel_to_3d(x, y, depth):
    X = (x - CX) * depth / FX
    Y = (y - CY) * depth / FY
    return np.array([X, Y, depth])

def get_file_start_time(filepath):
    """ファイル名（パス全体）から日時をパースする"""
    basename = os.path.basename(filepath)
    try:
        match = re.search(r'(\d{8})_(\d{6})', basename)
        if match:
            time_str = match.group(1) + "_" + match.group(2)
            return datetime.datetime.strptime(time_str, "%Y%m%d_%H%M%S")
    except Exception as e:
        print(f"タイムスタンプパースエラー ({basename}): {e}")
    
    return datetime.datetime.fromtimestamp(os.path.getmtime(filepath))

# 1. 階層を探索してRGBファイルのリストを再帰的に取得
rgb_pattern = os.path.join(BASE_DIR, "[0-9]*", "RGB", "[0-9]*", "*.mp4")
rgb_files = sorted(glob.glob(rgb_pattern))

print(f"検出されたRGBファイル数: {len(rgb_files)}本")

# 背景差分器の初期化
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# --- メイン解析ループ ---
for rgb_path in rgb_files:
    ir_path = rgb_path.replace(f"{os.sep}RGB{os.sep}", f"{os.sep}IR{os.sep}")
    
    if not os.path.exists(ir_path):
        print(f"警告: 対応するIRファイルが見つかりません。スキップします: {os.path.basename(rgb_path)}")
        continue
        
    print(f"解析中: {os.path.basename(rgb_path)}")
    
    cap_rgb = cv2.VideoCapture(rgb_path)
    cap_ir = cv2.VideoCapture(ir_path)
    
    file_start_time = get_file_start_time(rgb_path)
    prev_3d_pos = None
    frame_idx = 0
    
    while True:
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_ir, frame_ir = cap_ir.read()
        
        if not ret_rgb or not ret_ir:
            break
            
        h_rgb, w_rgb = frame_rgb.shape[:2]
        h_ir,  w_ir  = frame_ir.shape[:2]
        
        scale_x = w_ir / w_rgb
        scale_y = h_ir / h_rgb

        gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(frame_rgb)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx_ir = int(M["m10"] / M["m00"] * scale_x)
                    cy_ir = int(M["m01"] / M["m00"] * scale_y)
                    
                    cx_ir = max(0, min(cx_ir, w_ir - 1))
                    cy_ir = max(0, min(cy_ir, h_ir - 1))
                    
                    simulated_depth = (255 - gray_ir[cy_ir, cx_ir]) / 50.0 + 0.5 
                    
                    current_3d_pos = pixel_to_3d(cx_ir, cy_ir, simulated_depth)
                    
                    if prev_3d_pos is not None:
                        distance = np.linalg.norm(current_3d_pos - prev_3d_pos)
                        velocity = distance * FPS
                        momentum = OBJECT_MASS * velocity
                        
                        frame_time = file_start_time + datetime.timedelta(seconds=(frame_idx / FPS))
                        
                        with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([frame_time.strftime("%Y-%m-%d %H:%M:%S.%f"), momentum])
                            
                    prev_3d_pos = current_3d_pos
        
        frame_idx += 1
        
    cap_rgb.release()
    cap_ir.release()

print("全ファイルの解析が完了しました。1時間ごとに分割してグラフを出力します。")

# --- 3. CSVデータを1時間（Hour）ごとにグループ化してプロット ---
hourly_data = {}

with open(OUTPUT_CSV, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):
        # 描画軽量化のための間引き（10フレームに1点）
        if i % 10 == 0:
            dt = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
            val = float(row[1])
            
            # 「日付_時間」を辞書のキーにする (例: "20260603_15")
            hour_key = dt.strftime("%Y%m%d_%H")
            
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {"times": [], "values": []}
            
            hourly_data[hour_key]["times"].append(dt)
            hourly_data[hour_key]["values"].append(val)

# グループごとに個別グラフを出力
for hour_key, data in hourly_data.items():
    print(f"グラフ生成中: {hour_key}時台")
    
    plt.figure(figsize=(12, 4))
    plt.plot(data["times"], data["values"], color="crimson", linewidth=0.7)
    
    # グラフの見た目調整
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time")
    plt.ylabel("Momentum (kg*m/s)")
    plt.title(f"Momentum Plot - {hour_key}:00")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # 画像として保存 (例: momentum_plot_20260603_15.png)
    plt.savefig(f"momentum_plot_{hour_key}.png", dpi=150)
    plt.close()  # メモリ解放のために必ずクローズ

print("すべての時間帯のグラフ保存が完了しました。")
