# -*- coding: utf-8 -*-
"""
Intel RealSense L515 ― depth＋IR 動画保存スクリプト（拡張版）
- 起動時に「赤ちゃんID」「PC ID」を入力
- 各日の出力フォルダ(ROOT_PATH/YYYYMMDD/)に 000_RUN_INFO.txt を作成・追記
  （実行したコード名・画質・カメラ情報・入力IDなどを記録）
- HDF5属性に baby_id / pc_id / script も保存
"""

import os, sys, time, h5py, numpy as np, cv2 as cv, pyrealsense2 as rs
import platform
from datetime import datetime

# -------- ユーザ設定 --------
ROOT_PATH          = r"D:/Dev/Data"
W, H               = 1024, 768        # 1024 × 768
FPS                = 30               # LiDAR/IR は固定
SAVE_INTERVAL      = 6                # depthを約5fps化
FILE_PERIOD_MIN    = 1                # 1分で新ファイル
VISUALIZE          = True             # GUIプレビュー
PCT_CLIP           = 99               # depth→8bitクリップ率
INFO_FILENAME      = "000_RUN_INFO.txt"  # フォルダ先頭に来るよう命名
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

def open_h5(path, dscale, serial, baby_id, pc_id, script_name):
    cols = W * H
    f = h5py.File(path, "w")
    f.attrs.update({
        "depth_scale": dscale, "width": W, "height": H,
        "fps": FPS, "interval": SAVE_INTERVAL,
        "start_ts_sys": time.time(), "serial": serial,
        "baby_id": str(baby_id), "pc_id": str(pc_id), "script": str(script_name),
        "pct_clip": PCT_CLIP
    })
    d_depth = f.create_dataset("depth", (0, cols), maxshape=(None, cols),
                               dtype="uint16", chunks=(1, cols))
    d_ts    = f.create_dataset("ts", (0,), maxshape=(None,), dtype="float64")
    return f, d_depth, d_ts

def open_writer(path):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    return cv.VideoWriter(path, fourcc, FPS, (W, H), True)  # True=カラー(BGR)

# ---- ここがポイント：ウィンドウに“全体フィット（レターボックス）”して描画 ----
def show_fit(win_name, img):
    """
    現在のウィンドウ描画領域に画像を“全体が見切れない”ようにフィット表示。
    ウィンドウ比率と画像比率が違う場合は上下/左右に黒帯を入れる。
    """
    if not hasattr(cv, "getWindowImageRect"):
        cv.imshow(win_name, img); return

    x, y, win_w, win_h = cv.getWindowImageRect(win_name)
    if win_w < 2 or win_h < 2:
        win_w, win_h = img.shape[1], img.shape[0]

    ih, iw = img.shape[:2]
    scale = min(win_w / iw, win_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))

    interp = cv.INTER_NEAREST if new_w < iw or new_h < ih else cv.INTER_LINEAR
    resized = cv.resize(img, (new_w, new_h), interpolation=interp)

    if img.ndim == 2:
        canvas = np.zeros((win_h, win_w), dtype=np.uint8)
        y0 = (win_h - new_h) // 2; x0 = (win_w - new_w) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized
    else:
        canvas = np.zeros((win_h, win_w, img.shape[2]), dtype=np.uint8)
        y0 = (win_h - new_h) // 2; x0 = (win_w - new_w) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized

    cv.imshow(win_name, canvas)

def quality_string():
    approx_depth_fps = FPS / SAVE_INTERVAL if SAVE_INTERVAL else float(FPS)
    return (
        f"解像度: {W}x{H}\n"
        f"FPS: {FPS}\n"
        f"Depth保存: 1/{SAVE_INTERVAL} 間引き (≈ {approx_depth_fps:.2f} fps)\n"
        f"IR動画: MP4 (mp4v), {FPS} fps\n"
        f"Depth 8bit化 クリップ率: P{PCT_CLIP}\n"
    )

def script_name_string():
    # __file__ が無い環境でも安全に
    try:
        return os.path.basename(__file__)
    except NameError:
        return os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "interactive_session"

def write_run_info(base_path, baby_id, pc_id, script_name, dscale, serial):
    mkdir(base_path)
    info_path = os.path.join(base_path, INFO_FILENAME)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "==================== RUN ====================\n",
        f"時刻: {now}\n",
        f"赤ちゃんID: {baby_id}\n",
        f"PC ID: {pc_id}\n",
        f"実行スクリプト: {script_name}\n",
        "---- 画質設定 ----\n",
        quality_string(),
        "---- カメラ情報 ----\n",
        f"RealSense Serial: {serial}\n",
        f"Depth Scale: {dscale}\n",
        "=============================================\n\n",
    ]
    # 追記（同日で複数回起動しても履歴が残る）
    with open(info_path, "a", encoding="utf-8") as f:
        f.writelines(lines)

def main():
    # ---- 起動時入力 ----
    baby_id = input("赤ちゃんIDを入力してください: ").strip() or "unknown_baby"
    pc_id_in = input("PC IDを入力してください（空なら自動検出）: ").strip()
    pc_id = pc_id_in or os.environ.get("COMPUTERNAME") or platform.node() or "unknown_pc"
    script_name = script_name_string()

    mkdir(ROOT_PATH)
    try:
        pipe, dscale, serial = init_pipe()
    except Exception as e:
        sys.exit(f"パイプライン開始失敗: {e}")

    # ウォームアップ
    for _ in range(10):
        pipe.wait_for_frames()

    if VISUALIZE:
        cv.namedWindow("Depth-8bit", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Depth-8bit", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_FREERATIO)
        cv.resizeWindow("Depth-8bit", W // 2, H // 2)

        cv.namedWindow("IR", cv.WINDOW_NORMAL)
        cv.setWindowProperty("IR", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_FREERATIO)
        cv.resizeWindow("IR", W // 2, H // 2)

    frame_id = 0
    BLOCK_SECONDS = FILE_PERIOD_MIN * 60
    last_info_date = None  # 日付切り替えでINFOファイルを新規作成

    try:
        while True:
            now   = time.localtime()
            date  = time.strftime("%Y%m%d", now)
            hour  = time.strftime("%H", now)

            base_path  = os.path.join(ROOT_PATH, date)
            base_depth = os.path.join(base_path, "depth", hour)
            base_IR    = os.path.join(base_path, "IR",    hour)

            # ---- その日のフォルダに INFO を“先に”書く ----
            if last_info_date != date:
                write_run_info(base_path, baby_id, pc_id, script_name, dscale, serial)
                last_info_date = date

            mkdir(base_depth); mkdir(base_IR)

            prefix   = f"{date}_{hour}{now.tm_min:02d}{now.tm_sec:02d}"
            h5_path  = os.path.join(base_depth, f"{prefix}.h5")
            mp4_path = os.path.join(base_IR,    f"{prefix}.mp4")
            print("▶ 新ブロック:", prefix)

            h5, ds_depth, ds_ts = open_h5(h5_path, dscale, serial, baby_id, pc_id, script_name)
            writer = open_writer(mp4_path)
            saved_depth = 0
            block_start_time = time.time()

            while True:
                if time.time() - block_start_time > BLOCK_SECONDS:
                    break
                try:
                    frames = pipe.wait_for_frames(timeout_ms=4000)
                except RsErr:
                    continue

                dfrm = frames.get_depth_frame()
                ifrm = frames.get_infrared_frame()
                if not dfrm or not ifrm:
                    continue

                # --- depth を5fps保存 ---
                if frame_id % SAVE_INTERVAL == 0:
                    ds_depth.resize(ds_depth.shape[0]+1, 0)
                    ds_ts.resize(ds_ts.shape[0]+1, 0)
                    ds_depth[-1] = np.asarray(dfrm.get_data()).reshape(1, -1)
                    ds_ts[-1]    = dfrm.get_timestamp()
                    saved_depth += 1

                # --- IR を動画に追加 (30fps) ---
                ir_gray = np.asarray(ifrm.get_data(), dtype=np.uint8).reshape(H, W)
                writer.write(cv.cvtColor(ir_gray, cv.COLOR_GRAY2BGR))

                # --- 可視化 ---
                if VISUALIZE and frame_id % SAVE_INTERVAL == 0:
                    depth8 = depth_to_8bit(
                        np.asarray(dfrm.get_data(), dtype=np.uint16).reshape(H, W)
                    )
                    show_fit("Depth-8bit", depth8)
                    show_fit("IR", ir_gray)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                frame_id += 1

            # ---- ブロック終了 ----
            h5.attrs.update({"end_ts_sys": time.time(), "depth_frames": saved_depth})
            h5.close(); writer.release()
            print(f"▲ 保存完了: {h5_path} (depth {saved_depth}f) + {mp4_path}")

    except KeyboardInterrupt:
        print("\nユーザー停止")
    finally:
        if VISUALIZE: cv.destroyAllWindows()
        pipe.stop(); print("パイプライン停止完了")

if __name__ == "__main__":
    main()
