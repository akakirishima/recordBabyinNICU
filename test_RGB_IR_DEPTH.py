# -*- coding: utf-8 -*-
"""
Intel RealSense L515 ― Depth(H5) + IR + RGB 連続録画スクリプト（最適化・高信頼性版）

【ディレクトリ構成】
ROOT_PATH/babyID/DATE/Depth/<hour>/<prefix>.h5   (Resizable 3D配列 [Frame, H, W] / 高速LZF圧縮)
ROOT_PATH/babyID/DATE/IR/<hour>/<prefix>.mp4      (1024×768@30fps)
ROOT_PATH/babyID/DATE/RGB/<hour>/<prefix>.mp4     (1920×1080@30fps)
ROOT_PATH/babyID/DATE/<prefix>_info.txt           (セッション開始・日付変更時)
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
FILE_PERIOD_MIN  = 3                      # 何分ごとにファイル分割するか
VISUALIZE        = True                   # GUI プレビュー (描画負荷対策入り)
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

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth,    DEPTH_W, DEPTH_H, rs.format.z16,  FPS)
    cfg.enable_stream(rs.stream.infrared, IR_W,    IR_H,    rs.format.y8,   FPS)
    cfg.enable_stream(rs.stream.color,    RGB_W,   RGB_H,   rs.format.bgr8, FPS)

    cfg.resolve(rs.pipeline_wrapper(pipe))
    prof = pipe.start(cfg)

    dev = prof.get_device()
    serial = dev.get_info(rs.camera_info.serial_number)
    fw_ver = dev.get_info(rs.camera_info.firmware_version)
    dscale = dev.first_depth_sensor().get_depth_scale()
    return pipe, serial, fw_ver, dscale


def open_writer(path: str, width: int, height: int) -> cv.VideoWriter:
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(path, fourcc, FPS, (width, height), True)
    if not writer.isOpened():
        raise IOError(f"VideoWriterのオープンに失敗しました: {path}")
    return writer


def open_h5(path: str, dscale: float, serial: str):
    f = h5py.File(path, "w")
    # 後方解析で必須となるメタデータをHDF5内に埋め込み
    f.attrs.update({
        "depth_scale": dscale,
        "width": DEPTH_W,
        "height": DEPTH_H,
        "fps": FPS,
        "serial": serial,
    })
    # 形状は扱いやすい3次元(Frame, H, W)。圧縮は高速な "lzf" を採用してボトルネックを解消
    dset = f.create_dataset(
        "depth", 
        shape=(0, DEPTH_H, DEPTH_W), 
        maxshape=(None, DEPTH_H, DEPTH_W),
        dtype="uint16", 
        compression="lzf", 
        chunks=(1, DEPTH_H, DEPTH_W)
    )
    ts = f.create_dataset("ts", shape=(0,), maxshape=(None,), dtype="float64")
    return f, dset, ts

# ---------- メイン ----------

def main():
    mkdir(ROOT_PATH)

    try:
        pipe, serial, fw_ver, dscale = init_pipe()
    except Exception as e:
        sys.exit(f"パイプライン開始失敗: {e}")

    # AE（自動露出）安定のためのウォームアップ
    for _ in range(30):
        pipe.wait_for_frames()

    if VISUALIZE:
        cv.namedWindow("Depth (Visualized)", cv.WINDOW_NORMAL)
        cv.namedWindow("IR", cv.WINDOW_NORMAL)
        cv.namedWindow("RGB", cv.WINDOW_NORMAL)

    BLOCK_SECONDS = FILE_PERIOD_MIN * 60
    first_block = True
    last_info_date: str | None = None

    try:
        while True:
            now = time.localtime()
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

            # info.txt の生成（初回、または日付が切り替わったタイミング）
            if first_block or last_info_date != date_str:
                info_path = os.path.join(base, f"{prefix}_info.txt")
                with open(info_path, "w", encoding="utf-8") as f:
                    f.write(f"babyID    : {baby_id}\n")
                    f.write(f"pcname    : {pc_name}\n")
                    f.write(f"serial    : {serial}\n")
                    f.write(f"firmware  : {fw_ver}\n")
                    f.write(f"depth_res : {DEPTH_W}x{DEPTH_H}@{FPS}\n")
                    f.write(f"ir_res    : {IR_W}x{IR_H}@{FPS}\n")
                    f.write(f"rgb_res   : {RGB_W}x{RGB_H}@{FPS}\n")
                    f.write(f"start_iso : {datetime.now(timezone.utc).isoformat()}\n")
                    f.write(f"depth_comp: LZF\n")
                first_block = False
                last_info_date = date_str

            # ストリームライターのオープン
            ir_writer  = open_writer(mp4_ir,  IR_W,  IR_H)
            rgb_writer = open_writer(mp4_rgb, RGB_W, RGB_H)
            h5f, dset_depth, dset_ts = open_h5(h5_path, dscale, serial)

            block_start = time.monotonic()  # OS時刻変更に影響されない安全なタイマー
            frame_id = 0
            print(f"▶ 新ブロック録画開始: {prefix}")

            try:
                while time.monotonic() - block_start < BLOCK_SECONDS:
                    try:
                        frames = pipe.wait_for_frames(timeout_ms=3000)
                    except RsErr:
                        continue

                    dfrm = frames.get_depth_frame()
                    ifrm = frames.get_infrared_frame()
                    cfrm = frames.get_color_frame()
                    if not (dfrm and ifrm and cfrm):
                        continue

                    # データ配列の取得
                    depth_data = np.asanyarray(dfrm.get_data())     # (H, W) uint16
                    ir_gray    = np.asanyarray(ifrm.get_data())     # (H, W) uint8
                    rgb_img    = np.asanyarray(cfrm.get_data())     # (H, W, 3) BGR8

                    # --- 1. Depth 保存 (3次元拡張) ---
                    dset_depth.resize((frame_id + 1, DEPTH_H, DEPTH_W))
                    dset_ts.resize((frame_id + 1,))
                    dset_depth[frame_id] = depth_data
                    dset_ts[frame_id]    = dfrm.get_timestamp()

                    # --- 2. IR 保存 (MP4) ---
                    ir_writer.write(cv.cvtColor(ir_gray, cv.COLOR_GRAY2BGR))

                    # --- 3. RGB 保存 (MP4) ---
                    rgb_writer.write(rgb_img)

                    # --- 4. プレビュー (負荷制限付き) ---
                    if VISUALIZE and frame_id % 2 == 0:  # 描画は15fpsに間引いてCPUを保護
                        # 重いapplyColorMapではなく、高速なconvertScaleAbsで簡易可視化
                        depth_vis = cv.convertScaleAbs(depth_data, alpha=0.03)
                        cv.imshow("Depth (Visualized)", depth_vis)
                        cv.imshow("IR",  ir_gray)
                        cv.imshow("RGB", rgb_img)
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                    frame_id += 1

            finally:
                # リソースの確実な解放
                ir_writer.release()
                rgb_writer.release()
                h5f.close()
                print(f"▲ 保存完了: {prefix} (総フレーム数: {frame_id})")

    except KeyboardInterrupt:
        print("\nユーザーにより停止されました")
    finally:
        if VISUALIZE:
            cv.destroyAllWindows()
        pipe.stop()
        print("パイプライン停止完了")


if __name__ == "__main__":
    main()
