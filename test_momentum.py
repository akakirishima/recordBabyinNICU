# -*- coding: utf-8 -*-
"""
Intel RealSense L515 録画データ対応・ノイズ対策済 運動量解析スクリプト
（全動画データ統合 ＆ 1時間ごと完全グラフ化マネジメント版）
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
# ★解析したい「日付フォルダ」までを正確に指定してください
TARGET_DIR = r"D:/Dev/Data/babyID_example/20260608"

# 質量パラメータ
OBJECT_MASS = 1.0              # 物体の質量 (kg)
FILTER_WINDOW = 5              # 移動平均のウィンドウサイズ

# カメラの内部パラメータ (L515標準値)
FX, FY, CX, CY = 460.0, 460.0, 320.0, 240.0

def pixel_to_3d(x, y, depth_mm, scale):
    """ピクセル座標とミリメートル深度からメートル単位の3次元座標を計算"""
    depth_m = depth_mm * scale
    if depth_m <= 0.05:
        return None
    X = (x - CX) * depth_m / FX
    Y = (y - CY) * depth_m / FY
    return np.array([X, Y, depth_m])

# 1. 録画データの探索
rgb_pattern = os.path.join(TARGET_DIR, "RGB", "*", "*.mp4")
rgb_files = sorted(glob.glob(rgb_pattern))

print(f"検出されたRGBファイル数: {len(rgb_files)}本")
if not rgb_files:
    print(f"エラー: {TARGET_DIR} 内に動画ファイルが見つかりません。")
    exit()

# 背景差分器の初期化
fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# ★全動画のデータを一元管理するための巨大バッファ
all_timestamps = []
all_momentums = []

# --- メイン解析ループ（全動画からデータを抽出） ---
for rgb_path in rgb_files:
    video_basename = os.path.basename(rgb_path)
    
    # パス変換
    norm_path = rgb_path.replace("\\", "/")
    h5_path = norm_path.replace("/RGB/", "/Depth/").replace(".mp4", ".h5").replace("/", os.sep)
    
    if not os.path.exists(h5_path):
        print(f"警告: 対応するH5が見つかりません。スキップ: {video_basename}")
        continue
        
    print(f"データ抽出中: {video_basename}")
    
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
            
            device_ts_ms = ts_color_dset[frame_idx]
            frame_time = datetime.datetime.fromtimestamp(device_ts_ms / 1000.0)
            
            all_timestamps.append(frame_time)
            if contours and current_3d_pos is None and prev_3d_pos is not None:
                all_momentums.append(np.nan)
            else:
                all_momentums.append(momentum)
                if current_3d_pos is not None:
                    prev_3d_pos = current_3d_pos
            
            frame_idx += 1
            
    cap_rgb.release()

if not all_momentums:
    print("有効なデータが1件も解析されませんでした。")
    exit()

# --- 2. 視差スパイク・ノイズ対策アルゴリズム（全データ一括） ---
print("\n全データのクレンジングとスムージングを実行中...")
momentum_array = np.array(all_momentums, dtype=float)
nans = np.isnan(momentum_array)
if np.any(nans):
    x_indices = np.arange(len(momentum_array))
    momentum_array[nans] = np.interp(x_indices[nans], x_indices[~nans], momentum_array[~nans])

kernel_ma = np.ones(FILTER_WINDOW) / FILTER_WINDOW
smoothed_momentums = np.convolve(momentum_array, kernel_ma, mode='same')

# マスターCSVの保存（TARGET_DIR 直下）
output_csv_path = os.path.join(TARGET_DIR, "all_momentum_results.csv")
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Momentum(kg·m/s)"])
    for ts, val in zip(all_timestamps, smoothed_momentums):
        writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S.%f"), val])
print(f"マスターCSVを保存しました: {output_csv_path}")

# --- 3. 【大修正】データを1時間ごとに完全にグループ化してプロット ---
print("\n1時間ごとの統合グラフを生成中...")
hourly_data = {}
for i, ts in enumerate(all_timestamps):
    # グラフの描画が重くならないよう10フレームに1点間引き（必要なら 1 にすれば間引きなし）
    if i % 10 == 0:  
        hour_key = ts.strftime("%Y%m%d_%H")  # 例: "20260608_12"
        if hour_key not in hourly_data:
            hourly_data[hour_key] = {"times": [], "values": []}
        hourly_data[hour_key]["times"].append(ts)
        hourly_data[hour_key]["values"].append(smoothed_momentums[i])

# 1時間ごとに切り分けられたグループごとにグラフを出力
for hour_key, data in hourly_data.items():
    # 例: TARGET_DIR / "momentum_plot_20260608_12.png"
    graph_path = os.path.join(TARGET_DIR, f"momentum_plot_{hour_key}.png")
    
    plt.figure(figsize=(12, 4))
    plt.plot(data["times"], data["values"], color="crimson", linewidth=0.7)
    
    # グラフの見た目調整
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time")
    plt.ylabel("Momentum (kg*m/s)")
    plt.title(f"Noise-Filtered Momentum Plot - {hour_key[4:6]}/{hour_key[6:8]} {hour_key[9:]}:00")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"-> グラフを出力しました: {os.path.basename(graph_path)}")

print(f"\nすべての処理が完了しました。\nCSVおよび1時間ごとのグラフはすべて以下に保存されています：\n{TARGET_DIR}")
