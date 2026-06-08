# -*- coding: utf-8 -*-
"""
Intel RealSense L515 ― 高信頼性・非同期マルチスレッド 1800フレーム分割録画スクリプト
（2分無応答時のハイブリッド強制脱出安全弁付き）

【ディレクトリ構成】
ROOT_PATH/babyID/DATE/Depth/<hour>/<prefix>.h5
ROOT_PATH/babyID/DATE/IR/<hour>/<prefix>.mp4
ROOT_PATH/babyID/DATE/RGB/<hour>/<prefix>.mp4
ROOT_PATH/babyID/DATE/<prefix>_info.txt
"""

from __future__ import annotations
import os
import sys
import time
import queue
import threading
import numpy as np
import cv2 as cv
import h5py
import pyrealsense2 as rs

# -------- ユーザ設定 --------
ROOT_PATH        = r"D:/Dev/Data"         # データ保存先ルート
DEPTH_W, DEPTH_H = 1024, 768              # Depth 解像度
IR_W,    IR_H    = 1024, 768              # IR 解像度
RGB_W,   RGB_H   = 1920, 1080             # RGB 解像度 (負荷が高い場合は 1280, 720 に変更)
FPS              = 30                     # 共通フレームレート（L515は15または30のみ対応）
FRAMES_PER_FILE  = 1800                   # 1ファイルあたりの目標フレーム数 (30fpsで1分相当)
TIMEOUT_LIMIT    = 120                    # ★安全弁: カメラ無応答が2分（120秒）続いたら強制分割
VISUALIZE        = False                  # ★12時間連続撮影時は負荷軽減のため False を強く推奨
# ---------------------------

# ユーザー入力
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
        raise IOError(f"VideoWriterオープン失敗: {path}")
    return writer

# ---------- 非同期ディスク書き込みスレッド ----------
class AsyncFileWriter(threading.Thread):
    def __init__(self, prefix: str, base_dir: str, dscale: float, serial: str):
        super().__init__(daemon=True)
        self.prefix = prefix
        self.base_dir = base_dir
        self.q = queue.Queue(maxsize=150)  # 約5秒分のセーフティバッファ
        self.running = True

        # パス構築
        self.h5_path  = os.path.join(base_dir, "Depth", prefix[9:11], f"{prefix}.h5")
        self.mp4_ir   = os.path.join(base_dir, "IR",    prefix[9:11], f"{prefix}.mp4")
        self.mp4_rgb  = os.path.join(base_dir, "RGB",   prefix[9:11], f"{prefix}.mp4")
        
        for p in (os.path.dirname(self.h5_path), os.path.dirname(self.mp4_ir), os.path.dirname(self.mp4_rgb)):
            mkdir(p)

        # 各種ライターの準備
        self.ir_writer = open_writer(self.mp4_ir, IR_W, IR_H)
        self.rgb_writer = open_writer(self.mp4_rgb, RGB_W, RGB_H)
        
        self.h5f = h5py.File(self.h5_path, "w")
        self.h5f.attrs.update({
            "depth_scale": dscale, "width": DEPTH_W, "height": DEPTH_H, "fps": FPS, "serial": serial,
        })
        self.dset_depth = self.h5f.create_dataset(
            "depth", shape=(0, DEPTH_H, DEPTH_W), maxshape=(None, DEPTH_H, DEPTH_W),
            dtype="uint16", compression="lzf", chunks=(1, DEPTH_H, DEPTH_W)
        )
        self.ts_depth = self.h5f.create_dataset("ts_depth", shape=(0,), maxshape=(None,), dtype="float64")
        self.ts_color = self.h5f.create_dataset("ts_color", shape=(0,), maxshape=(None,), dtype="float64")
        self.ts_infra = self.h5f.create_dataset("ts_infra", shape=(0,), maxshape=(None,), dtype="float64")

        self.frame_id = 0

    def run(self):
        while self.running or not self.q.empty():
            try:
                data = self.q.get(timeout=0.1)
                depth, ir, rgb, ts_d, ts_c, ts_i = data
                
                # 1. HDF5への書き込み（データ領域の動的拡張）
                self.dset_depth.resize((self.frame_id + 1, DEPTH_H, DEPTH_W))
                self.ts_depth.resize((self.frame_id + 1,))
                self.ts_color.resize((self.frame_id + 1,))
                self.ts_infra.resize((self.frame_id + 1,))

                self.dset_depth[self.frame_id] = depth
                self.ts_depth[self.frame_id]   = ts_d
                self.ts_color[self.frame_id]   = ts_c
                self.ts_infra[self.frame_id]   = ts_i

                # 2. ビデオエンコード処理
                self.ir_writer.write(cv.cvtColor(ir, cv.COLOR_GRAY2BGR))
                self.rgb_writer.write(rgb)

                # 定期フラッシュ（クラッシュ時のデータ破損最小化）
                if self.frame_id % FPS == 0:
                    self.h5f.flush()

                self.frame_id += 1
                self.q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n[Error] 書き込みエラー: {e}", file=sys.stderr)
                break
        self.close_resources()

    def close_resources(self):
        try: self.ir_writer.release()
        except: pass
        try: self.rgb_writer.release()
        except: pass
        try:
            self.h5f.flush()
            self.h5f.close()
        except: pass
        print(f"▲ 保存完了: {self.prefix} (総記録フレーム数: {self.frame_id})")


# ---------- メインループ ----------
def main():
    mkdir(ROOT_PATH)

    try:
        pipe, serial, fw_ver, dscale = init_pipe()
    except Exception as e:
        sys.exit(f"パイプライン開始失敗: {e}")

    # カメラ初期化後の自動露出安定待ち（設定FPSに依存）
    for _ in range(FPS):
        pipe.wait_for_frames()

    if VISUALIZE:
        cv.namedWindow("Depth (Visualized)", cv.WINDOW_NORMAL)
        cv.namedWindow("IR", cv.WINDOW_NORMAL)
        cv.namedWindow("RGB", cv.WINDOW_NORMAL)

    first_block = True
    last_info_date: str | None = None
    writer_thread: AsyncFileWriter | None = None

    try:
        while True:
            now = time.localtime()
            date_str = time.strftime("%Y%m%d", now)
            hour_str = time.strftime("%H",      now)
            base_dir = os.path.join(ROOT_PATH, baby_id, date_str)
            mkdir(base_dir)

            prefix = f"{date_str}_{hour_str}{now.tm_min:02d}{now.tm_sec:02d}"

            # ログ用メタデータテキストの出力
            if first_block or last_info_date != date_str:
                info_path = os.path.join(base_dir, f"{prefix}_info.txt")
                with open(info_path, "w", encoding="utf-8") as f:
                    f.write(f"babyID    : {baby_id}\n")
                    f.write(f"pcname    : {pc_name}\n")
                    f.write(f"serial    : {serial}\n")
                    f.write(f"firmware  : {fw_ver}\n")
                    f.write(f"depth_comp: LZF (Async / Hybrid Frame Split)\n")
                first_block = False
                last_info_date = date_str

            # 書き込みスレッドの生成と開始
            writer_thread = AsyncFileWriter(prefix, base_dir, dscale, serial)
            writer_thread.start()
            print(f"▶ ブロック録画開始 (目標:{FRAMES_PER_FILE}枚): {prefix}")

            local_frame_count = 0
            last_frame_time = time.monotonic()  # タイムアウト監視用

            # フレームキャプチャループ
            while local_frame_count < FRAMES_PER_FILE:
                
                # 【安全弁判定】前回のフレーム取得成功から設定時間（2分）以上空いたらループ脱出
                if time.monotonic() - last_frame_time > TIMEOUT_LIMIT:
                    print(f"\n[CRITICAL ERROR] カメラ応答が {TIMEOUT_LIMIT} 秒間途絶えたため、"
                          f"ファイル「{prefix}」を強制終了して次へ移行します。", file=sys.stderr)
                    break

                try:
                    # 瞬断を即時検知するためタイムアウトは500msに設定
                    frames = pipe.wait_for_frames(timeout_ms=500)
                except RsErr:
                    continue

                dfrm = frames.get_depth_frame()
                ifrm = frames.get_infrared_frame()
                cfrm = frames.get_color_frame()
                if not (dfrm and ifrm and cfrm):
                    continue

                # フレームが取得できたら生存時間を更新
                last_frame_time = time.monotonic()

                # メタデータ用デバイスタイムスタンプ
                ts_d = dfrm.get_timestamp()
                ts_c = cfrm.get_timestamp()
                ts_i = ifrm.get_timestamp()

                # numpy配列へ変換
                depth_data = np.asanyarray(dfrm.get_data())
                ir_gray    = np.asanyarray(ifrm.get_data())
                rgb_img    = np.asanyarray(cfrm.get_data())

                # 非同期キューへ投入
                try:
                    writer_thread.q.put_nowait((depth_data, ir_gray, rgb_img, ts_d, ts_c, ts_i))
                except queue.Full:
                    print("[Warning] キュー満杯によるコマ落ち発生！", file=sys.stderr)

                # プレビュー表示（負荷軽減のため2フレームに1回間引き）
                if VISUALIZE and local_frame_count % 2 == 0:
                    depth_vis = cv.convertScaleAbs(depth_data, alpha=0.03)
                    cv.imshow("Depth (Visualized)", depth_vis)
                    cv.imshow("IR",  ir_gray)
                    cv.imshow("RGB", rgb_img)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                local_frame_count += 1

            # ループを抜けたら、現在のスレッドに終了命令を出し、書き込み完了を同期して待つ
            writer_thread.running = False
            writer_thread.join()
            writer_thread = None

    except KeyboardInterrupt:
        print("\nユーザーにより停止されました。")
    finally:
        if writer_thread is not None:
            writer_thread.running = False
            writer_thread.join()
        if VISUALIZE:
            cv.destroyAllWindows()
        try:
            pipe.stop()
            print("パイプライン停止完了")
        except:
            pass

if __name__ == "__main__":
    main()
