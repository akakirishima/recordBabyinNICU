# -*- coding: utf-8 -*-
"""
Intel RealSense L515 — Depth + IR + RGB プレビュー (録画と完全一致版)
  • Depth : 1024×768 @30 fps  (Z16 → 8-bit GRAY, 1倍スケール)
  • IR    : 1024×768 @30 fps  (1ch GRAY)
  • RGB   : 1280×720 @30 fps  (BGR)
  • 'q' で終了
"""

from __future__ import annotations
import sys, numpy as np, cv2 as cv, pyrealsense2 as rs

# -------- ユーザ設定 (録画スクリプトと同じ値に合わせる) --------
DEPTH_W, DEPTH_H  = 1024, 768
IR_W,    IR_H     = 1024, 768
RGB_W,   RGB_H    = 1280, 720
DEPTH_FPS = IR_FPS = RGB_FPS = 30
QUEUE_SIZE        = 16
# --------------------------------------------------------------

# ---------- RealSense 初期化 ----------
ctx = rs.context()
if not ctx.devices:
    sys.exit("L515 が見つかりません。")
dev  = ctx.devices[0]
pipe = rs.pipeline()

cfg = rs.config()
cfg.enable_stream(rs.stream.depth,    DEPTH_W, DEPTH_H, rs.format.z16,  DEPTH_FPS)
cfg.enable_stream(rs.stream.infrared, IR_W,    IR_H,    rs.format.y8,   IR_FPS)
# RGB を BGR フォーマットで取得する
cfg.enable_stream(rs.stream.color,    RGB_W,   RGB_H,   rs.format.bgr8, RGB_FPS)

# 内部フレームキューサイズを統一
for s in dev.query_sensors():
    if s.supports(rs.option.frames_queue_size):
        s.set_option(rs.option.frames_queue_size, QUEUE_SIZE)

pipe.start(cfg)

# 自動露光安定化のため数十フレーム捨てる
for _ in range(30):
    pipe.wait_for_frames()

print("=== PREVIEW START (press 'q' to quit) ===")

try:
    while True:
        frames = pipe.wait_for_frames(timeout_ms=2000)
        dfrm = frames.get_depth_frame()
        ifrm = frames.get_infrared_frame()
        cfrm = frames.get_color_frame()
        if not (dfrm and ifrm and cfrm):
            continue

        # Depth: 16-bit → 8-bit グレースケール (線形)
        depth = np.asanyarray(dfrm.get_data(), dtype=np.uint16)
        depth_8u = (depth >> 8).astype(np.uint8)
        depth_vis = cv.cvtColor(depth_8u, cv.COLOR_GRAY2BGR)

        # IR (1ch) → 3ch 揃え
        ir_img = np.asanyarray(ifrm.get_data())  # uint8
        ir_vis = cv.cvtColor(ir_img, cv.COLOR_GRAY2BGR)

        # RGB (すでに BGR 順)
        rgb_vis = np.asanyarray(cfrm.get_data())  # (H,W,3) uint8

        # ウィンドウ表示（録画と同サイズそのまま）
        cv.imshow("Depth (RAW-GRAY)", depth_vis)
        cv.imshow("IR",              ir_vis)
        cv.imshow("RGB",             rgb_vis)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv.destroyAllWindows()
    print("=== PREVIEW STOP ===")
