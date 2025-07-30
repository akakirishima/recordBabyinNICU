# -*- coding: utf-8 -*-
"""
Intel RealSense L515 ― depth＋IR 動画保存スクリプト
取得        : 640×480 @ 30 fps
Depth保存   : 6フレームに1回（約5fps）をHDF5
IR保存      : 30fps そのままMP4
分割        : 3分ごとにHDF5とMP4を同名で生成（depthとIRはサブフォルダで分離）
"""

import os, sys, time, h5py, numpy as np, cv2 as cv, pyrealsense2 as rs

# -------- ユーザ設定 --------
ROOT_PATH          = r"D:/Dev/Data"
W, H               = 1024, 768        # 1024 × 768
FPS                = 30               # LiDAR IR は固定
SAVE_INTERVAL      = 6                # depthを5fps化
FILE_PERIOD_MIN    = 1                # 1分で新ファイル
VISUALIZE          = True             # GUIプレビュー
PCT_CLIP           = 99               # depth→8bitクリップ率
# ----------------------------

RsErr = rs.error if hasattr(rs, "error") else Exception
def mkdir(p): os.makedirs(p, exist_ok=True)

def depth_to_8bit(d16):
    valid = d16[d16 > 0]
    vmax  = np.percentile(valid, PCT_CLIP) if valid.size else 1
    d8 = np.clip(d16.astype(np.float32), 0, vmax) / vmax * 255
    return d8.astype(np.uint8)

def init_pipe():
    ctx = rs.context()
    if not ctx.devices: raise RuntimeError("L515 が接続されていません。")
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth,    W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.infrared, W, H, rs.format.y8,  FPS)
    cfg.resolve(rs.pipeline_wrapper(pipe))
    prof = pipe.start(cfg)
    dscale = prof.get_device().first_depth_sensor().get_depth_scale()
    serial = prof.get_device().get_info(rs.camera_info.serial_number)
    return pipe, dscale, serial

def open_h5(path, dscale, serial):
    cols = W * H
    f = h5py.File(path, "w")
    f.attrs.update({"depth_scale":dscale,"width":W,"height":H,
                    "fps":FPS,"interval":SAVE_INTERVAL,
                    "start_ts_sys":time.time(),"serial":serial})
    d_depth = f.create_dataset("depth",(0,cols),maxshape=(None,cols),
                               dtype="uint16",chunks=(1,cols))
    d_ts    = f.create_dataset("ts",(0,),maxshape=(None,),dtype="float64")
    return f, d_depth, d_ts

def open_writer(path):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    return cv.VideoWriter(path, fourcc, FPS, (W, H), True)  # True=カラー

def main():
    mkdir(ROOT_PATH)
    try:
        pipe, dscale, serial = init_pipe()
    except Exception as e:
        sys.exit(f"パイプライン開始失敗: {e}")

    # ウォームアップ
    for _ in range(10):
        pipe.wait_for_frames()
    if VISUALIZE:
        cv.namedWindow("Depth‑8bit"); cv.namedWindow("IR")

    frame_id = 0
    BLOCK_SECONDS = FILE_PERIOD_MIN * 60

    try:
        while True:
            now   = time.localtime()
            date  = time.strftime("%Y%m%d", now)
            hour  = time.strftime("%H", now)
            # ----------- 日付の直下に depth と IR フォルダを作成し、さらに時間で分ける ----------
            base_path = os.path.join(ROOT_PATH, date)
            base_depth  = os.path.join(base_path, "depth", hour)
            base_IR     = os.path.join(base_path, "IR", hour)
            mkdir(base_depth)
            mkdir(base_IR)
            prefix = f"{date}_{hour}{now.tm_min:02d}{now.tm_sec:02d}"
            h5_path  = os.path.join(base_depth, f"{prefix}.h5")
            mp4_path = os.path.join(base_IR,    f"{prefix}.mp4")
            print("▶ 新ブロック:", prefix)

            h5, ds_depth, ds_ts = open_h5(h5_path, dscale, serial)
            writer = open_writer(mp4_path)
            saved_depth = 0
            block_start_time = time.time()

            while True:
                if time.time() - block_start_time > BLOCK_SECONDS:
                    break
                try:
                    frames = pipe.wait_for_frames(timeout_ms=4000)
                except RsErr: continue

                dfrm = frames.get_depth_frame()
                ifrm = frames.get_infrared_frame()
                if not dfrm or not ifrm: continue

                # --- depth を5fps保存 ---
                if frame_id % SAVE_INTERVAL == 0:
                    ds_depth.resize(ds_depth.shape[0]+1,0)
                    ds_ts.resize(ds_ts.shape[0]+1,0)
                    ds_depth[-1] = np.asarray(dfrm.get_data()).reshape(1,-1)
                    ds_ts[-1]    = dfrm.get_timestamp()
                    saved_depth += 1

                # --- IR を動画に追加 (30fps) ---
                ir_gray = np.asarray(ifrm.get_data(), dtype=np.uint8).reshape(H,W)
                writer.write(cv.cvtColor(ir_gray, cv.COLOR_GRAY2BGR))

                # --- 可視化 ---
                if VISUALIZE and frame_id % SAVE_INTERVAL == 0:
                    depth8 = depth_to_8bit(np.asarray(dfrm.get_data(), dtype=np.uint16).reshape(H,W))
                    cv.imshow("Depth‑8bit", depth8)
                    cv.imshow("IR", ir_gray)
                    if cv.waitKey(1) & 0xFF == ord('q'): raise KeyboardInterrupt

                frame_id += 1

            # ---- ブロック終了 ----
            h5.attrs.update({"end_ts_sys":time.time(),"depth_frames":saved_depth})
            h5.close(); writer.release()
            print(f"▲ 保存完了: {h5_path} (depth {saved_depth}f) + {mp4_path}")

    except KeyboardInterrupt:
        print("\nユーザー停止")
    finally:
        if VISUALIZE: cv.destroyAllWindows()
        pipe.stop(); print("パイプライン停止完了")

if __name__ == "__main__":
    main()
