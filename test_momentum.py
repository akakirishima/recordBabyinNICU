# -*- coding: utf-8 -*-
"""
Intel RealSense L515 録画データ対応・ノイズ対策済 運動量解析スクリプト
（視差スパイク対策：0値線形補完 ＆ 移動平均フィルター搭載）
"""

import os
import glob
import datetime
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import csv
import h5py

# --- 設定 ---
BASE_DIR = r"D:/Dev/Data/babyID_example/20260608" 
OUTPUT_CSV = "momentum_results.csv"
OBJECT_MASS = 1.0              # 物体の質量 (kg)

# ノイズ対策用の設定パラメータ
FILTER_WINDOW = 5              # 移動平均のウィンドウサイズ（5フレーム＝約0.16秒間の平滑化）

# カメラの内部パラメータ (L515標準値)
FX, FY, CX, CY = 460.0, 460.0, 320.0, 240.0

def pixel_to_3d(x, y, depth_mm, scale):
    """ピクセル座標とミリメートル深度からメートル単位の3次元座標を計算"""
    depth_m = depth_mm * scale
    if depth_m <= 0.05: # あまりに近すぎる、または0（無効値）は排除
        return None
    X = (x - CX) * depth_m / FX
    Y = (y - CY) * depth_m / FY
    return np.array([X, Y, depth_m])

# 1. 録画データの取得
rgb_pattern = os.path.join(BASE_DIR, "RGB", "*", "*.mp4")
rgb_files = sorted(glob.glob(rgb_pattern))

print(f"検出されたRGBファイル数: {len(rgb_files)}本")
if not rgb_files:
    print("指定されたディレクトリに動画ファイルが見つかりません。")
    exit()

# 背景差分器の初期化
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# 解析結果を一時的に保持するリスト
raw_timestamps = []
raw_momentums = []

# --- メイン解析ループ ---
for rgb_path in rgb_files:
    norm_path = rgb_path.replace("\\", "/")
    h5_path = norm_path.replace("/RGB/", "/Depth/").replace(".mp4", ".h5").replace("/", os.sep)
    
    if not os.path.exists(h5_path):
        continue
        
    print(f"データ抽出中: {os.path.basename(rgb_path)}")
    
    cap_rgb = cv.VideoCapture(rgb_path)
    fps = cap_rgb.get(cv.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
        
    w_rgb = int(cap_rgb.get(cv.CAP_PROP_FRAME_WIDTH))
    h_rgb = int(cap_rgb.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    prev_3d_pos = None
    frame_idx = 0
    
    with h5py.File(h5_path, "r") as h5f:
        dscale = h5f.attrs.get("depth_scale", 0.00025)
        depth_dset = h5f["depth"]
        ts_color_dset = h5f["ts_color"]
        total_h5_frames = depth_dset.shape[0]
        
        while True:
            ret_rgb, frame_rgb = cap_rgb.read()
            if not ret_rgb or frame_idx >= total_h5_frames:
                break
                
            depth_frame = depth_dset[frame_idx]
            h_depth, w_depth = depth_frame.shape[:2]
            
            scale_x = w_depth / w_rgb
            scale_y = h_depth / h_rgb
            
            fgmask = fgbg.apply(frame_rgb)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            
            contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # 初期値としてこのフレームの運動量は0、位置はNoneで進める
            momentum = 0.0
            current_3d_pos = None
            
            if contours:
                largest_contour = max(contours, key=cv.contourArea)
                if cv.contourArea(largest_contour) > 500:
                    M = cv.moments(largest_contour)
                    if M["m00"] != 0:
                        cx_depth = int(M["m10"] / M["m00"] * scale_x)
                        cy_depth = int(M["m01"] / M["m00"] * scale_y)
                        cx_depth = max(0, min(cx_depth, w_depth - 1))
                        cy_depth = max(0, min(cy_depth, h_depth - 1))
                        
                        real_depth_val = depth_frame[cy_depth, cx_depth]
                        current_3d_pos = pixel_to_3d(cx_depth, cy_depth, real_depth_val, dscale)
                        
                        if current_3d_pos is not None and prev_3d_pos is not None:
                            distance = np.linalg.norm(current_3d_pos - prev_3d_pos)
                            velocity = distance * fps
                            momentum = OBJECT_MASS * velocity
            
            # タイムスタンプの取得と記録
            device_ts_ms = ts_color_dset[frame_idx]
            frame_time = datetime.datetime.fromtimestamp(device_ts_ms / 1000.0)
            
            raw_timestamps.append(frame_time)
            # 輪郭エッジの誤サンプリング（Depth=0や異常な距離ジャンプ）は一旦「NaN」か「0」にして保持
            if contours and current_3d_pos is None and prev_3d_pos is not None:
                # 視差ズレによる突発ノイズの可能性が高いため、無効値(NaN)としてマーク
                raw_momentums.append(np.nan)
            else:
                raw_momentums.append(momentum)
                if current_3d_pos is not None:
                    prev_3d_pos = current_3d_pos
            
            frame_idx += 1
            
    cap_rgb.release()

if not raw_momentums:
    print("有効なデータが解析されませんでした。")
    exit()

# --- 2. 視差スパイク・ノイズ対策アルゴリズム（信号処理） ---
print("データクレンジング中（ノイズフィルタ処理）...")
momentum_array = np.array(raw_momentums, dtype=float)

# 対策 A: 視差ズレで発生した NaN（無効データ）を前後の線形補完で埋める
nans = np.isnan(momentum_array)
if np.any(nans):
    # numpyのインターポレートでNaNを補完
    x_indices = np.arange(len(momentum_array))
    momentum_array[nans] = np.interp(x_indices[nans], x_indices[~nans], momentum_array[~nans])

# 対策 B: 移動平均フィルターによるスムージング処理（突発スパイクの平滑化）
kernel_ma = np.ones(FILTER_WINDOW) / FILTER_WINDOW
smoothed_momentums = np.convolve(momentum_array, kernel_ma, mode='same')

# クレンジング後のデータをCSVへ保存
print(f"CSVファイルへ {len(smoothed_momentums)} 件の平滑化データを書き込み中...")
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Momentum(kg·m/s)"])
    for ts, val in zip(raw_timestamps, smoothed_momentums):
        writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S.%f"), val])

# --- 3. CSVデータを1時間（Hour）ごとにグループ化してプロット ---
hourly_data = {}
for i, ts in enumerate(raw_timestamps):
    if i % 10 == 0:  # 描画軽量化
        hour_key = ts.strftime("%Y%m%d_%H")
        if hour_key not in hourly_data:
            hourly_data[hour_key] = {"times": [], "values": []}
        hourly_data[hour_key]["times"].append(ts)
        hourly_data[hour_key]["values"].append(smoothed_momentums[i])

# グループごとに個別グラフを出力
for hour_key, data in hourly_data.items():
    print(f"グラフ生成中: {hour_key}時台")
    plt.figure(figsize=(12, 4))
    plt.plot(data["times"], data["values"], color="crimson", linewidth=0.7)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time")
    plt.ylabel("Momentum (kg*m/s)")
    plt.title(f"Noise-Filtered Momentum Plot - {hour_key}:00")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"momentum_plot_{hour_key}.png", dpi=150)
    plt.close()

print("すべての処理が完了しました。")
