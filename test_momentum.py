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
        # 正規表現で「YYYYMMDD_HHMMSS」のパターンを抽出
        match = re.search(r'(\d{8})_(\d{6})', basename)
        if match:
            time_str = match.group(1) + "_" + match.group(2)
            return datetime.datetime.strptime(time_str, "%Y%m%d_%H%M%S")
    except Exception as e:
        print(f"タイムスタンプパースエラー ({basename}): {e}")
    
    # 失敗時はファイルの修正日時
    return datetime.datetime.fromtimestamp(os.path.getmtime(filepath))

# 1. 階層を探索してRGBファイルのリストを再帰的に取得
# 例: ./20260603/RGB/15/20260603_155043.mp4
rgb_pattern = os.path.join(BASE_DIR, "[0-9]*", "RGB", "[0-9]*", "*.mp4")
rgb_files = sorted(glob.glob(rgb_pattern))

print(f"検出されたRGBファイル数: {len(rgb_files)}本")

# 背景差分器の初期化（ループの外側で行い、12時間連続で学習させる）
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# --- メイン解析ループ ---
for rgb_path in rgb_files:
    # 2. RGBのパスから対応するIRのパスを動的に生成
    # 例: .../RGB/... -> .../IR/...
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
        
        # RGBからIRへ座標変換するための倍率
        scale_x = w_ir / w_rgb
        scale_y = h_ir / h_rgb


        gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(frame_rgb)
        
        # モルフォロジー演算によるノイズ除去
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
                    
                    # 配列の範囲外（IndexError）を防ぐための安全弁（クリッピング）
                    cx_ir = max(0, min(cx_ir, w_ir - 1))
                    cy_ir = max(0, min(cy_ir, h_ir - 1))
                    
                    # IR輝度からの仮の深度計算（実環境に合わせてキャリブレーションが必要）
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

print("全ファイルの解析が完了しました。グラフを描画します。")

# --- 3. CSVからデータを読み込んで間引き描画（メモリ対策） ---
times = []
momentums = []
with open(OUTPUT_CSV, mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):
        # 130万点をすべて描画すると重いため、10フレームに1点（約0.3秒ごと）に間引く
        if i % 10 == 0:
            times.append(datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f"))
            momentums.append(float(row[1]))

plt.figure(figsize=(15, 5))
plt.plot(times, momentums, color="crimson", linewidth=0.5)
plt.gcf().autofmt_xdate()
plt.xlabel("Absolute Time")
plt.ylabel("Momentum (kg·m/s)")
plt.title("12-Hour Momentum Plot (Downsampled)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
