# -*- coding: utf-8 -*-
"""
Intel RealSense L515 — Depth+IR 30 fps / RGB 6 fps 録画スクリプト
  • Depth : 1024×768 @30 fps → .h5（1800 枚）
  • IR    : 1024×768 @30 fps → .avi（MJPG・GRAY 1 ch）
  • RGB   :  960×540 @ 6 fps → .mp4
  • 60 s ごとにファイル分割
"""

from __future__ import annotations
import os, sys, time, numpy as np, cv2 as cv, h5py, pyrealsense2 as rs
from datetime import datetime, timezone

# -------- ユーザ設定 --------
ROOT_PATH         = r"D:/Dev/Data"
DEPTH_W, DEPTH_H  = 1024, 768
IR_W,    IR_H     = 1024, 768
RGB_W,   RGB_H    = 1280, 720
DEPTH_FPS = IR_FPS = 30
RGB_FPS            = 6
BLOCK_SEC          = 60
VISUALIZE          = False
QUEUE_SIZE         = 32
# ---------------------------

# ==== babyID / PCname ====
baby_id = input("Enter baby ID   : ").strip()
pc_name = input("Enter PC name   : ").strip()

ctx = rs.context()
if not ctx.devices:
    sys.exit("L515 が接続されていません。")
dev = ctx.devices[0]
for s in dev.query_sensors():
    if s.supports(rs.option.frames_queue_size):
        s.set_option(rs.option.frames_queue_size, QUEUE_SIZE)

# --- Depth / IR pipeline ---
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
cfg.enable_stream(rs.stream.infrared, IR_W, IR_H, rs.format.y8, IR_FPS)
prof = pipe.start(cfg)
serial   = dev.get_info(rs.camera_info.serial_number)
firmware = dev.get_info(rs.camera_info.firmware_version)
dscale   = dev.first_depth_sensor().get_depth_scale()

# --- RGB sensor 6 fps ---
color_sensor = next(s for s in dev.query_sensors()
                    if s.get_info(rs.camera_info.name) == "RGB Camera")
rgb_prof = next(p for p in color_sensor.get_stream_profiles()
                if p.fps() == RGB_FPS and
                   p.as_video_stream_profile().width()  == RGB_W and
                   p.as_video_stream_profile().height() == RGB_H)
q_rgb = rs.frame_queue(QUEUE_SIZE)
color_sensor.open(rgb_prof)
color_sensor.start(lambda f: q_rgb.enqueue(f))

# AE 安定
for _ in range(30): pipe.wait_for_frames()

# ---------- ヘルパ ----------
def mkdir(p): os.makedirs(p, exist_ok=True)

def writer_ir(path: str):
    """IR を GRAY+MJPG で保存（isColor=False）"""
    vw = cv.VideoWriter(path,
                        cv.VideoWriter_fourcc(*"MJPG"),
                        IR_FPS, (IR_W, IR_H), False)
    if not vw.isOpened(): raise IOError(f"open {path}")
    return vw

def writer_rgb(path: str):
    vw = cv.VideoWriter(path,
                        cv.VideoWriter_fourcc(*"mp4v"),
                        RGB_FPS, (RGB_W, RGB_H), True)
    if not vw.isOpened(): raise IOError(f"open {path}")
    return vw

def open_h5(path: str):
    cols = DEPTH_W * DEPTH_H
    f = h5py.File(path, "w")
    f.attrs.update({"depth_scale": dscale, "width": DEPTH_W, "height": DEPTH_H,
                    "depth_fps": DEPTH_FPS, "ir_fps": IR_FPS,
                    "rgb_fps": RGB_FPS, "serial": serial})
    dset = f.create_dataset("depth", (DEPTH_FPS*BLOCK_SEC, cols),
                            dtype="uint16", chunks=(64, cols))
    ts   = f.create_dataset("ts", (DEPTH_FPS*BLOCK_SEC,), dtype="float64")
    return f, dset, ts

# ---------- 録画ループ ----------
while True:
    now = time.localtime()
    date_str, hour_str = time.strftime("%Y%m%d", now), time.strftime("%H", now)
    base = os.path.join(ROOT_PATH, baby_id, date_str)
    ddir = os.path.join(base, "Depth", hour_str)
    idir = os.path.join(base, "IR",    hour_str)
    rdir = os.path.join(base, "RGB",   hour_str)
    for p in (ddir, idir, rdir): mkdir(p)

    prefix  = f"{date_str}_{hour_str}{now.tm_min:02d}{now.tm_sec:02d}"
    h5_path = os.path.join(ddir, f"{prefix}.h5")
    avi_ir  = os.path.join(idir, f"{prefix}.avi")      # GRAY+MJPG
    mp4_rgb = os.path.join(rdir, f"{prefix}.mp4")

    # info.txt は同日初回のみ
    info_txt = os.path.join(base, f"{prefix}_info.txt")
    if not os.path.exists(info_txt):
        with open(info_txt, "w", encoding="utf-8") as f:
            f.write(f"babyID  : {baby_id}\npcname  : {pc_name}\n")
            f.write(f"serial  : {serial}\nfirmware: {firmware}\n")
            f.write(f"depth   : {DEPTH_W}x{DEPTH_H}@{DEPTH_FPS}\n")
            f.write(f"ir      : {IR_W}x{IR_H}@{IR_FPS}\n")
            f.write(f"rgb     : {RGB_W}x{RGB_H}@{RGB_FPS}\n")
            f.write(f"start   : {datetime.now(timezone.utc).isoformat()}\n")

    ir_w  = writer_ir(avi_ir)
    rgb_w = writer_rgb(mp4_rgb)
    h5f, dset, ts = open_h5(h5_path)

    depth_idx = ir_count = rgb_count = 0
    print("▶ 新ブロック:", prefix)
    try:
        while depth_idx < DEPTH_FPS * BLOCK_SEC:
            try:
                fs = pipe.wait_for_frames(timeout_ms=2000)
            except rs.error: continue
            dfrm, ifrm = fs.get_depth_frame(), fs.get_infrared_frame()
            if not (dfrm and ifrm): continue

            # Depth → HDF5
            depth = np.asanyarray(dfrm.get_data(), dtype=np.uint16)
            dset[depth_idx] = depth.reshape(-1)
            ts[depth_idx]   = dfrm.get_timestamp()
            depth_idx += 1

            # IR → MJPG (GRAY)
            ir_img = np.asanyarray(ifrm.get_data())  # uint8 (H,W)
            ir_w.write(ir_img); ir_count += 1

            # RGB → mp4
            rgb_frame = q_rgb.poll_for_frame()
            if rgb_frame and rgb_count < RGB_FPS*BLOCK_SEC:
                rgb_w.write(np.asanyarray(rgb_frame.get_data())); rgb_count += 1

            # プレビュー（任意）
            if VISUALIZE and depth_idx % 30 == 0:
                cv.imshow("Depth", cv.convertScaleAbs(depth, alpha=0.03))
                cv.imshow("IR", ir_img)
                if rgb_count: cv.imshow("RGB", np.asanyarray(rgb_frame.get_data()))
                if cv.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
    finally:
        h5f.close(); ir_w.release(); rgb_w.release()
        print(f"▲ 保存完了: Depth={depth_idx}  IR={ir_count}  RGB={rgb_count}")

# ---- Ctrl-C で停止 ----
