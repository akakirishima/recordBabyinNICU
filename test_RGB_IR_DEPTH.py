# -*- coding: utf-8 -*-
"""
Intel RealSense L515 — IR + RGB + Depth 連続録画・撮影スクリプト

ディレクトリ構成
ROOT_PATH/babyID/DATE/IR/<hour>/*.mp4
ROOT_PATH/babyID/DATE/RGB/<hour>/*.mp4
ROOT_PATH/babyID/DATE/Depth/<hour>/*.npz (各フレームの16bit深度データを圧縮保存)
ROOT_PATH/babyID/DATE/*_info.txt  ← 録画セッション開始時に 1 ファイルのみ生成

IR    : 1024 × 768 @ 30 fps (L515 最大)
RGB   : 1920 × 1080 @ 30 fps (L515 最大)
Depth : 1024 × 768 @ 30 fps (IRと同期)
分割  : 1 分ごとにファイル群を自動生成
必須入力 : babyID と PCname（スクリプト起動時プロンプト）
"""

import os
import sys
import time
import numpy as np
import cv2 as cv
import pyrealsense2 as rs

# -------- ユーザ設定 --------
ROOT_PATH       = r"D:/Dev/Data"    # データ保存先ルート
IR_W,  IR_H      = 1024, 768         # IR 解像度 (Max)
RGB_W, RGB_H     = 1920, 1080        # RGB 解像度 (Max)
DEPTH_W, DEPTH_H = 1024, 768         # Depth 解像度 (IRと一致させる)
FPS              = 30                # 共通フレームレート
FILE_PERIOD_MIN  = 1                 # 1分ごとにファイル分割
VISUALIZE        = True              # GUI プレビュー
# ---------------------------

# ==== babyID / PCname を必須入力 ====
baby_id = ""
pc_name = ""
while not baby_id.strip():
    baby_id = input("Enter baby ID   : ").strip()
while not pc_name.strip():
    pc_name = input("Enter PC name   : ").strip()

RsErr = rs.error if hasattr(rs, "error") else Exception

def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def init_pipe():
    ctx = rs.context()
    if not ctx.devices:
        raise RuntimeError("L515 が接続されていません。")

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.infrared, IR_W, IR_H, rs.format.y8, FPS)
    cfg.enable_stream(rs.stream.color,    RGB_W, RGB_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth,    DEPTH_W, DEPTH_H, rs.format.z16, FPS)

    cfg.resolve(rs.pipeline_wrapper(pipe))
    prof = pipe.start(cfg)

    serial = prof.get_device().get_info(rs.camera_info.serial_number)
    return pipe, serial


def open_writer(path: str, width: int, height: int):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    return cv.VideoWriter(path, fourcc, FPS, (width, height), True)


def main():
    mkdir(ROOT_PATH)

    try:
        pipe, serial = init_pipe()
    except Exception as e:
        sys.exit(f"パイプライン開始失敗: {e}")

    # オートエクスポージャ・レーザー安定用ウォームアップ
    for _ in range(30):
        pipe.wait_for_frames()

    if VISUALIZE:
        cv.namedWindow("IR")
        cv.namedWindow("RGB")
        cv.namedWindow("Depth (Visualized)")

    BLOCK_SECONDS = FILE_PERIOD_MIN * 60
    first_block   = True  # info.txt を最初のブロックだけ生成するフラグ

    try:
        while True:
            now   = time.localtime()
            date  = time.strftime("%Y%m%d", now)  # 例 20260605
            hour  = time.strftime("%H",      now)  # 例 11

            # 各種保存フォルダの生成
            base_path  = os.path.join(ROOT_PATH, baby_id, date)
            base_IR    = os.path.join(base_path, "IR",    hour)
            base_RGB   = os.path.join(base_path, "RGB",   hour)
            base_Depth = os.path.join(base_path, "Depth", hour)
            
            for p in (base_IR, base_RGB, base_Depth):
                mkdir(p)

            prefix = f"{date}_{hour}{now.tm_min:02d}{now.tm_sec:02d}"
            mp4_ir_path  = os.path.join(base_IR,  f"{prefix}.mp4")
            mp4_rgb_path = os.path.join(base_RGB, f"{prefix}.mp4")
            
            # Depthは各ファイル群をまとめるサブディレクトリをプレフィックス名で作る
            depth_block_dir = os.path.join(base_Depth, prefix)
            mkdir(depth_block_dir)

            # info.txt はスクリプト起動後、最初のブロックのみ生成
            if first_block:
                info_path = os.path.join(base_path, f"{prefix}_info.txt")
                with open(info_path, "w", encoding="utf-8") as f:
                    f.write(f"babyID   : {baby_id}\n")
                    f.write(f"pcname   : {pc_name}\n")
                    f.write(f"serial   : {serial}\n")
                    f.write(f"start_ts_sys : {time.time()}\n")
                first_block = False

            ir_writer  = open_writer(mp4_ir_path,  IR_W,  IR_H)
            rgb_writer = open_writer(mp4_rgb_path, RGB_W, RGB_H)
            
            block_start = time.time()
            print(f"▶ 新ブロック ({FILE_PERIOD_MIN}分録画開始): {prefix}")

            frame_count = 0
            try:
                while time.time() - block_start <= BLOCK_SECONDS:
                    try:
                        frames = pipe.wait_for_frames(timeout_ms=3000)
                    except RsErr:
                        continue

                    ifrm = frames.get_infrared_frame()
                    cfrm = frames.get_color_frame()
                    dfrm = frames.get_depth_frame()
                    
                    if not ifrm or not cfrm or not dfrm:
                        continue

                    # 1. IRデータの処理・書き込み
                    ir_gray = np.asanyarray(ifrm.get_data()).reshape(IR_H, IR_W)
                    ir_writer.write(cv.cvtColor(ir_gray, cv.COLOR_GRAY2BGR))

                    # 2. RGBデータの処理・書き込み
                    rgb_img = np.asanyarray(cfrm.get_data())
                    rgb_writer.write(rgb_img)

                    # 3. Depthデータの処理・書き込み (16bit値をそのままNPZ圧縮保存)
                    depth_data = np.asanyarray(dfrm.get_data()).reshape(DEPTH_H, DEPTH_W)
                    depth_frame_path = os.path.join(depth_block_dir, f"{frame_count:06d}.npz")
                    np.savez_compressed(depth_frame_path, depth=depth_data)

                    # 4. プレビュー表示
                    if VISUALIZE:
                        # Depthは可視化用にカラーマップを適用 (表示用であり、データ保存は16bitのまま)
                        depth_colored = cv.applyColorMap(
                            cv.convertScaleAbs(depth_data, alpha=0.03), cv.COLORMAP_JET
                        )
                        cv.imshow("IR",  ir_gray)
                        cv.imshow("RGB", rgb_img)
                        cv.imshow("Depth (Visualized)", depth_colored)
                        
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                    frame_count += 1

            finally:
                ir_writer.release()
                rgb_writer.release()
                print(f"▲ 保存完了: {prefix} (総フレーム数: {frame_count})")

    except KeyboardInterrupt:
        print("\nユーザー停止")
    finally:
        if VISUALIZE:
            cv.destroyAllWindows()
        pipe.stop()
        print("パイプライン停止完了")


if __name__ == "__main__":
    main()
