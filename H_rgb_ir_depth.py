# -*- coding: utf-8 -*-
"""
Intel RealSense L515 ― Depth + IR + RGB 連続録画スクリプト（最大画質版）

ディレクトリ構成
ROOT_PATH/babyID/DATE/Depth/<hour>/<prefix>.h5   (16‑bit Z16 全フレーム)
ROOT_PATH/babyID/DATE/IR/<hour>/<prefix>.mp4     (1024×768@30 fps)
ROOT_PATH/babyID/DATE/RGB/<hour>/<prefix>.mp4    (1920×1080@30 fps)
ROOT_PATH/babyID/DATE/<prefix>_info.txt          (セッション開始時のみ)

すべて **最大解像度 × 30 fps** で記録し、Depth は可逆 HDF5（無圧縮）に保存します。
記録開始時に babyID / PCname を入力し、デバイス情報・ストリーム構成を info.txt に出力します。
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np
import cv2 as cv
import h5py
import pyrealsense2 as rs
from datetime import datetime, timezone

# -------- ユーザ設定 --------
ROOT_PATH        = r"D:/Dev/Data"         # データ保存先ルート
DEPTH_W, DEPTH_H = 1024, 768              # Depth 解像度 (Max)
IR_W,    IR_H    = 1024, 768              # IR   解像度 (Max)
RGB_W,   RGB_H   = 1920, 1080             # RGB  解像度 (Max)
FPS              = 30                     # 共通フレームレート
FILE_PERIOD_MIN  = 1                      # 何分ごとにファイル分割
VISUALIZE        = False                  # GUI プレビュー
# ---------------------------

# ==== babyID / PCname を必須入力 ====
baby_id = ""
pc_name = ""
while not baby_id.strip():
    baby_id = input("Enter baby ID   : ").strip()
while not pc_name.strip():
    pc_name = input("Enter PC name   : ").strip()

RsErr = rs.error if hasattr(rs, "error") else Exception

# ---------- ヘルパ ----------

def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def init_pipe():
    ctx = rs.context()
    if not ctx.devices:
        raise RuntimeError("L515 が接続されていません。")

    dev = ctx.devices[0]
    firmware = dev.get_info(rs.camera_info.firmware_version)

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth,    DEPTH_W, DEPTH_H, rs.format.z16,  FPS)
    cfg.enable_stream(rs.stream.infrared, IR_W,    IR_H,    rs.format.y8,   FPS)
    cfg.enable_stream(rs.stream.color,    RGB_W,   RGB_H,   rs.format.bgr8, FPS)

    cfg.resolve(rs.pipeline_wrapper(pipe))
    prof = pipe.start(cfg)

    serial = prof.get_device().get_info(rs.camera_info.serial_number)
    dscale = prof.get_device().first_depth_sensor().get_depth_scale()
    return pipe, serial, firmware, dscale


def open_writer(path: str, width: int, height: int) -> cv.VideoWriter:
    fourcc = cv.VideoWriter_fourcc(*"mp4v")  # 環境依存。必要なら 'avc1' などへ変更
    writer = cv.VideoWriter(path, fourcc, FPS, (width, height), True)
    if not writer.isOpened():
        raise IOError(f"VideoWriter open failed: {path}")
    return writer


def open_h5(path: str, dscale: float, serial: str):
    cols = DEPTH_W * DEPTH_H
    f = h5py.File(path, "w")
    f.attrs.update({
        "depth_scale": dscale,
        "width": DEPTH_W, "height": DEPTH_H,
        "fps": FPS,
        "serial": serial,
    })
    dset = f.create_dataset("depth", (0, cols), maxshape=(None, cols),
                           dtype="uint16", chunks=(1, cols))
    ts   = f.create_dataset("ts", (0,), maxshape=(None,), dtype="float64")
    return f, dset, ts

# ---------- メイン ----------

def main():
    mkdir(ROOT_PATH)

    try:
        pipe, serial, fw_ver, dscale = init_pipe()
    except Exception as e:
        sys.exit(f"パイプライン開始失敗: {e}")

    # AE 安定
    for _ in range(30):
        pipe.wait_for_frames()

    if VISUALIZE:
        cv.namedWindow("Depth8"); cv.namedWindow("IR"); cv.namedWindow("RGB")

    BLOCK_SECONDS = FILE_PERIOD_MIN * 60
    first_block   = True
    last_info_date: str | None = None

    try:
        while True:
            now      = time.localtime()
            date_str = time.strftime("%Y%m%d", now)
            hour_str = time.strftime("%H",      now)

            # フォルダ構築
            base = os.path.join(ROOT_PATH, baby_id, date_str)
            dir_depth = os.path.join(base, "Depth", hour_str)
            dir_ir    = os.path.join(base, "IR",    hour_str)
            dir_rgb   = os.path.join(base, "RGB",   hour_str)
            for p in (dir_depth, dir_ir, dir_rgb):
                mkdir(p)

            prefix = f"{date_str}_{hour_str}{now.tm_min:02d}{now.tm_sec:02d}"
            h5_path   = os.path.join(dir_depth, f"{prefix}.h5")
            mp4_ir    = os.path.join(dir_ir,    f"{prefix}.mp4")
            mp4_rgb   = os.path.join(dir_rgb,   f"{prefix}.mp4")

            # info.txt → 日付切替 or 初回のみ
            if first_block or last_info_date != date_str:
                info_path = os.path.join(base, f"{prefix}_info.txt")
                with open(info_path, "w", encoding="utf-8") as f:
                    f.write(f"babyID  : {baby_id}\n")
                    f.write(f"pcname  : {pc_name}\n")
                    f.write(f"serial  : {serial}\n")
                    f.write(f"firmware: {fw_ver}\n")
                    f.write(f"depth_res : {DEPTH_W}x{DEPTH_H}@{FPS}\n")
                    f.write(f"ir_res    : {IR_W}x{IR_H}@{FPS}\n")
                    f.write(f"rgb_res   : {RGB_W}x{RGB_H}@{FPS}\n")
                    f.write(f"start_iso : {datetime.now(timezone.utc).isoformat()}\n")
                first_block = False
                last_info_date = date_str

            # --- Writer / HDF5 open ---
            ir_writer  = open_writer(mp4_ir,  IR_W,  IR_H)
            rgb_writer = open_writer(mp4_rgb, RGB_W, RGB_H)
            h5f, dset_depth, dset_ts = open_h5(h5_path, dscale, serial)

            block_start = time.monotonic()
            frame_id    = 0
            print("▶ 新ブロック:", prefix)

            try:
                while time.monotonic() - block_start <= BLOCK_SECONDS:
                    try:
                        frames = pipe.wait_for_frames(timeout_ms=3000)
                    except RsErr:
                        continue

                    dfrm = frames.get_depth_frame()
                    ifrm = frames.get_infrared_frame()
                    cfrm = frames.get_color_frame()
                    if not (dfrm and ifrm and cfrm):
                        continue

                    # --- Depth 保存 (全フレーム) ---
                    depth = np.asanyarray(dfrm.get_data())      # (H,W) uint16
                    dset_depth.resize(dset_depth.shape[0] + 1, 0)
                    dset_ts.resize(dset_ts.shape[0] + 1, 0)
                    dset_depth[-1] = depth.reshape(1, -1)
                    dset_ts[-1]    = dfrm.get_timestamp()

                    # --- IR 保存 (MP4) ---
                    ir_gray = np.asanyarray(ifrm.get_data())    # (H,W) uint8
                    ir_writer.write(cv.cvtColor(ir_gray, cv.COLOR_GRAY2BGR))

                    # --- RGB 保存 (MP4) ---
                    rgb_img = np.asanyarray(cfrm.get_data())    # (H,W,3) uint8
                    rgb_writer.write(rgb_img)

                    # --- プレビュー ---
                    if VISUALIZE and frame_id % 2 == 0:
                        depth8 = cv.convertScaleAbs(depth, alpha=0.03)  # 粗 8‑bit 表示
                        cv.imshow("Depth8", depth8)
                        cv.imshow("IR",     ir_gray)
                        cv.imshow("RGB",    rgb_img)
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt
                    frame_id += 1

            finally:
                h5f.close(); ir_writer.release(); rgb_writer.release()
                print(f"▲ 保存完了: {h5_path} + {mp4_ir} + {mp4_rgb}")

    except KeyboardInterrupt:
        print("\nユーザー停止")
    finally:
        if VISUALIZE:
            cv.destroyAllWindows()
        pipe.stop(); print("パイプライン停止完了")


if __name__ == "__main__":
    main()
