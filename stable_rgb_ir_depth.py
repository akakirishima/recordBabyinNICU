# -*- coding: utf-8 -*-
"""
Intel RealSense L515 - stable-first RGB + IR + Depth recorder.

Goal:
    Reduce frame loss risk by favoring RGB, then IR, then Depth.

Default profile:
    RGB   : 1280x720  @ 30 fps -> MP4
    IR    : 1024x768  @ 15 fps -> MP4
    Depth : 1024x768  @  5 fps save target -> HDF5

Design notes:
    - Acquisition and disk writes are separated with per-stream queues.
    - Under backpressure, Depth is skipped first, then IR.
    - RGB is only dropped if its own queue is full.
    - Output structure matches the existing repository layout.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import cv2 as cv
import h5py
import numpy as np
import pyrealsense2 as rs

# -------- User settings --------
ROOT_PATH = r"D:/Dev/Data"

RGB_W, RGB_H = 1280, 720
RGB_FPS = 30

IR_W, IR_H = 1024, 768
IR_FPS = 15

DEPTH_W, DEPTH_H = 1024, 768
DEPTH_FPS_SAVE = 5

FILE_PERIOD_MIN = 1
VISUALIZE = False

QUEUE_SIZE_RGB = 90
QUEUE_SIZE_IR = 45
QUEUE_SIZE_DEPTH = 15
# ------------------------------

WAIT_TIMEOUT_MS = 2000
WARMUP_FRAMES = 30
QUEUE_PRESSURE_SKIP_DEPTH = 0.70
QUEUE_PRESSURE_SKIP_IR = 0.90
RsErr = rs.error if hasattr(rs, "error") else Exception


@dataclass
class StreamCounters:
    acquired: int = 0
    enqueued: int = 0
    saved: int = 0
    dropped_queue: int = 0
    skipped_policy: int = 0
    skipped_sampling: int = 0


class SessionStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.streams = {
            "rgb": StreamCounters(),
            "ir": StreamCounters(),
            "depth": StreamCounters(),
        }
        self.blocks: list[dict[str, Any]] = []

    def increment(self, stream: str, field_name: str, amount: int = 1) -> None:
        with self._lock:
            counters = self.streams[stream]
            setattr(counters, field_name, getattr(counters, field_name) + amount)

    def add_block(self, block_info: dict[str, Any]) -> None:
        with self._lock:
            self.blocks.append(block_info)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            streams = {name: asdict(counters) for name, counters in self.streams.items()}
            blocks = list(self.blocks)
        return {"streams": streams, "blocks": blocks}


class DepthSampler:
    def __init__(self, capture_fps: int, save_fps: int) -> None:
        if capture_fps <= 0 or save_fps <= 0:
            raise ValueError("Depth FPS must be positive.")
        if save_fps > capture_fps:
            raise ValueError("DEPTH_FPS_SAVE cannot exceed the depth capture FPS.")
        self._credit = 0.0
        self._ratio = save_fps / capture_fps

    def should_save(self) -> bool:
        self._credit += self._ratio
        if self._credit + 1e-9 >= 1.0:
            self._credit -= 1.0
            return True
        return False


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompt_required(prompt: str) -> str:
    value = ""
    while not value.strip():
        value = input(prompt).strip()
    return value


def queue_load(q: queue.Queue[Any]) -> float:
    if q.maxsize <= 0:
        return 0.0
    return q.qsize() / q.maxsize


def should_skip_depth(
    rgb_queue: queue.Queue[Any],
    ir_queue: queue.Queue[Any],
    depth_queue: queue.Queue[Any],
) -> bool:
    return max(queue_load(rgb_queue), queue_load(ir_queue), queue_load(depth_queue)) >= QUEUE_PRESSURE_SKIP_DEPTH


def should_skip_ir(
    rgb_queue: queue.Queue[Any],
    ir_queue: queue.Queue[Any],
) -> bool:
    return max(queue_load(rgb_queue), queue_load(ir_queue)) >= QUEUE_PRESSURE_SKIP_IR


def write_json(path: str, data: dict[str, Any]) -> None:
    mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_info_txt(path: str, session: dict[str, Any], device: dict[str, Any]) -> None:
    cfg = session["config"]
    lines = [
        f"session_id : {session['session_id']}",
        f"baby_id    : {session['baby_id']}",
        f"pc_name    : {session['pc_name']}",
        f"start_iso  : {session['start_iso']}",
        "",
        "[device]",
        f"serial     : {device['serial']}",
        f"firmware   : {device['firmware']}",
        f"depth_scale: {device['depth_scale']}",
        "",
        "[config]",
        f"root_path        : {cfg['root_path']}",
        f"rgb              : {cfg['rgb']['width']}x{cfg['rgb']['height']} @ {cfg['rgb']['fps']} fps",
        f"ir               : {cfg['ir']['width']}x{cfg['ir']['height']} @ {cfg['ir']['fps']} fps",
        f"depth_capture    : {cfg['depth']['width']}x{cfg['depth']['height']} @ {cfg['depth']['capture_fps']} fps",
        f"depth_save       : {cfg['depth']['save_fps']} fps",
        f"file_period_min  : {cfg['file_period_min']}",
        f"visualize        : {cfg['visualize']}",
        f"queue_rgb        : {cfg['queue_sizes']['rgb']}",
        f"queue_ir         : {cfg['queue_sizes']['ir']}",
        f"queue_depth      : {cfg['queue_sizes']['depth']}",
        "",
        "[priority]",
        "order            : RGB > IR > Depth",
        "policy           : skip Depth first, then IR, keep RGB last",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_session_summary(
    path: str,
    session: dict[str, Any],
    device: dict[str, Any],
    stats: SessionStats,
    status: str,
    error_message: str | None = None,
) -> None:
    payload = {
        "status": status,
        "error": error_message,
        **session,
        "device": device,
        **stats.snapshot(),
        "updated_iso": utc_now_iso(),
    }
    write_json(path, payload)


def build_block_paths(root_path: str, baby_id: str) -> dict[str, str]:
    now = time.localtime()
    date_str = time.strftime("%Y%m%d", now)
    hour_str = time.strftime("%H", now)
    prefix = f"{date_str}_{hour_str}{now.tm_min:02d}{now.tm_sec:02d}"

    base = os.path.join(root_path, baby_id, date_str)
    rgb_dir = os.path.join(base, "RGB", hour_str)
    ir_dir = os.path.join(base, "IR", hour_str)
    depth_dir = os.path.join(base, "Depth", hour_str)
    for path in (base, rgb_dir, ir_dir, depth_dir):
        mkdir(path)

    return {
        "block_id": prefix,
        "date": date_str,
        "hour": hour_str,
        "base": base,
        "rgb_path": os.path.join(rgb_dir, f"{prefix}.mp4"),
        "ir_path": os.path.join(ir_dir, f"{prefix}.mp4"),
        "depth_path": os.path.join(depth_dir, f"{prefix}.h5"),
        "started_iso": utc_now_iso(),
    }


def open_video_writer(path: str, width: int, height: int, fps: int) -> cv.VideoWriter:
    writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    if not writer.isOpened():
        raise IOError(f"VideoWriter open failed: {path}")
    return writer


def open_depth_h5(path: str, session: dict[str, Any], device: dict[str, Any]):
    cols = DEPTH_W * DEPTH_H
    f = h5py.File(path, "w")
    f.attrs.update(
        {
            "session_id": session["session_id"],
            "baby_id": session["baby_id"],
            "pc_name": session["pc_name"],
            "serial": device["serial"],
            "firmware": device["firmware"],
            "depth_scale": device["depth_scale"],
            "width": DEPTH_W,
            "height": DEPTH_H,
            "capture_fps": session["config"]["depth"]["capture_fps"],
            "save_fps": session["config"]["depth"]["save_fps"],
        }
    )
    depth_dset = f.create_dataset("depth", (0, cols), maxshape=(None, cols), dtype="uint16", chunks=(1, cols))
    ts_dset = f.create_dataset("ts", (0,), maxshape=(None,), dtype="float64")
    return f, depth_dset, ts_dset


def configure_sensor_queues(device: rs.device) -> None:
    sensor_queue_size = max(QUEUE_SIZE_RGB, QUEUE_SIZE_IR, QUEUE_SIZE_DEPTH)
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.frames_queue_size):
            sensor.set_option(rs.option.frames_queue_size, sensor_queue_size)


def init_pipeline() -> tuple[rs.pipeline, dict[str, Any]]:
    ctx = rs.context()
    if not ctx.devices:
        raise RuntimeError("L515 が接続されていません。")

    device = ctx.devices[0]
    configure_sensor_queues(device)

    depth_capture_fps = max(IR_FPS, DEPTH_FPS_SAVE)
    firmware = device.get_info(rs.camera_info.firmware_version)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, depth_capture_fps)
    cfg.enable_stream(rs.stream.infrared, IR_W, IR_H, rs.format.y8, IR_FPS)
    cfg.enable_stream(rs.stream.color, RGB_W, RGB_H, rs.format.bgr8, RGB_FPS)

    try:
        cfg.resolve(rs.pipeline_wrapper(pipe))
        profile = pipe.start(cfg)
    except Exception as exc:
        raise RuntimeError(f"パイプライン開始失敗: {exc}") from exc

    started_device = profile.get_device()
    info = {
        "serial": started_device.get_info(rs.camera_info.serial_number),
        "firmware": firmware,
        "depth_scale": started_device.first_depth_sensor().get_depth_scale(),
        "depth_capture_fps": depth_capture_fps,
    }
    return pipe, info


def push_frame(
    frame_queue: queue.Queue[Any],
    item: dict[str, Any],
    stats: SessionStats,
    stream_name: str,
) -> None:
    try:
        frame_queue.put_nowait(item)
        stats.increment(stream_name, "enqueued")
    except queue.Full:
        stats.increment(stream_name, "dropped_queue")


def rgb_writer(
    frame_queue: queue.Queue[Any],
    stop_event: threading.Event,
    error_queue: queue.Queue[Any],
    stats: SessionStats,
) -> None:
    writer: cv.VideoWriter | None = None
    current_block: str | None = None
    try:
        while True:
            item = frame_queue.get()
            if item is None:
                break
            if item["block_id"] != current_block:
                if writer is not None:
                    writer.release()
                writer = open_video_writer(item["path"], RGB_W, RGB_H, RGB_FPS)
                current_block = item["block_id"]
            writer.write(item["frame"])
            stats.increment("rgb", "saved")
    except Exception as exc:
        error_queue.put(("rgb", exc))
        stop_event.set()
    finally:
        if writer is not None:
            writer.release()


def ir_writer(
    frame_queue: queue.Queue[Any],
    stop_event: threading.Event,
    error_queue: queue.Queue[Any],
    stats: SessionStats,
) -> None:
    writer: cv.VideoWriter | None = None
    current_block: str | None = None
    try:
        while True:
            item = frame_queue.get()
            if item is None:
                break
            if item["block_id"] != current_block:
                if writer is not None:
                    writer.release()
                writer = open_video_writer(item["path"], IR_W, IR_H, IR_FPS)
                current_block = item["block_id"]
            writer.write(cv.cvtColor(item["frame"], cv.COLOR_GRAY2BGR))
            stats.increment("ir", "saved")
    except Exception as exc:
        error_queue.put(("ir", exc))
        stop_event.set()
    finally:
        if writer is not None:
            writer.release()


def depth_writer(
    frame_queue: queue.Queue[Any],
    stop_event: threading.Event,
    error_queue: queue.Queue[Any],
    stats: SessionStats,
    session: dict[str, Any],
    device: dict[str, Any],
) -> None:
    h5f: h5py.File | None = None
    dset_depth = None
    dset_ts = None
    current_block: str | None = None
    try:
        while True:
            item = frame_queue.get()
            if item is None:
                break
            if item["block_id"] != current_block:
                if h5f is not None:
                    h5f.close()
                h5f, dset_depth, dset_ts = open_depth_h5(item["path"], session, device)
                current_block = item["block_id"]
            depth = item["frame"]
            dset_depth.resize(dset_depth.shape[0] + 1, 0)
            dset_ts.resize(dset_ts.shape[0] + 1, 0)
            dset_depth[-1] = depth.reshape(1, -1)
            dset_ts[-1] = item["timestamp"]
            stats.increment("depth", "saved")
    except Exception as exc:
        error_queue.put(("depth", exc))
        stop_event.set()
    finally:
        if h5f is not None:
            h5f.close()


def put_stop_signal(frame_queue: queue.Queue[Any], worker: threading.Thread) -> None:
    while worker.is_alive():
        try:
            frame_queue.put(None, timeout=0.2)
            return
        except queue.Full:
            if not worker.is_alive():
                return


def preview_frames(depth_frame: np.ndarray | None, ir_frame: np.ndarray | None, rgb_frame: np.ndarray | None) -> bool:
    if depth_frame is not None:
        cv.imshow("Depth", cv.convertScaleAbs(depth_frame, alpha=0.03))
    if ir_frame is not None:
        cv.imshow("IR", ir_frame)
    if rgb_frame is not None:
        cv.imshow("RGB", rgb_frame)
    return (cv.waitKey(1) & 0xFF) == ord("q")


def main() -> None:
    mkdir(ROOT_PATH)

    baby_id = prompt_required("Enter baby ID   : ")
    pc_name = prompt_required("Enter PC name   : ")

    pipe: rs.pipeline | None = None
    stop_event = threading.Event()
    stats = SessionStats()
    error_queue: queue.Queue[Any] = queue.Queue()
    rgb_queue: queue.Queue[Any] = queue.Queue(maxsize=QUEUE_SIZE_RGB)
    ir_queue: queue.Queue[Any] = queue.Queue(maxsize=QUEUE_SIZE_IR)
    depth_queue: queue.Queue[Any] = queue.Queue(maxsize=QUEUE_SIZE_DEPTH)

    worker_threads: list[tuple[threading.Thread, queue.Queue[Any]]] = []
    session_started_local = time.localtime()
    session_id = time.strftime("%Y%m%d_%H%M%S", session_started_local)
    session_day = time.strftime("%Y%m%d", session_started_local)
    session_base = os.path.join(ROOT_PATH, baby_id, session_day)
    mkdir(session_base)

    error_message: str | None = None
    status = "running"

    try:
        pipe, device = init_pipeline()

        session = {
            "session_id": session_id,
            "baby_id": baby_id,
            "pc_name": pc_name,
            "start_iso": utc_now_iso(),
            "config": {
                "root_path": ROOT_PATH,
                "rgb": {"width": RGB_W, "height": RGB_H, "fps": RGB_FPS},
                "ir": {"width": IR_W, "height": IR_H, "fps": IR_FPS},
                "depth": {
                    "width": DEPTH_W,
                    "height": DEPTH_H,
                    "capture_fps": device["depth_capture_fps"],
                    "save_fps": DEPTH_FPS_SAVE,
                },
                "file_period_min": FILE_PERIOD_MIN,
                "visualize": VISUALIZE,
                "queue_sizes": {
                    "rgb": QUEUE_SIZE_RGB,
                    "ir": QUEUE_SIZE_IR,
                    "depth": QUEUE_SIZE_DEPTH,
                },
                "priority_order": ["rgb", "ir", "depth"],
            },
        }
        info_path = os.path.join(session_base, f"{session_id}_info.txt")
        summary_path = os.path.join(session_base, f"{session_id}_summary.json")
        write_info_txt(info_path, session, device)
        write_session_summary(summary_path, session, device, stats, status="starting")

        depth_sampler = DepthSampler(device["depth_capture_fps"], DEPTH_FPS_SAVE)

        rgb_thread = threading.Thread(target=rgb_writer, args=(rgb_queue, stop_event, error_queue, stats), daemon=True)
        ir_thread = threading.Thread(target=ir_writer, args=(ir_queue, stop_event, error_queue, stats), daemon=True)
        depth_thread = threading.Thread(
            target=depth_writer,
            args=(depth_queue, stop_event, error_queue, stats, session, device),
            daemon=True,
        )
        worker_threads = [
            (rgb_thread, rgb_queue),
            (ir_thread, ir_queue),
            (depth_thread, depth_queue),
        ]
        for thread, _ in worker_threads:
            thread.start()

        for _ in range(WARMUP_FRAMES):
            pipe.wait_for_frames()

        if VISUALIZE:
            cv.namedWindow("RGB")
            cv.namedWindow("IR")
            cv.namedWindow("Depth")

        block_seconds = FILE_PERIOD_MIN * 60
        block_deadline = 0.0
        current_block: dict[str, str] | None = None

        print("=== STABLE RECORD START ===")
        while not stop_event.is_set():
            if not error_queue.empty():
                stream_name, exc = error_queue.get_nowait()
                raise RuntimeError(f"{stream_name} writer failed: {exc}") from exc

            now_mono = time.monotonic()
            if current_block is None or now_mono >= block_deadline:
                current_block = build_block_paths(ROOT_PATH, baby_id)
                block_deadline = now_mono + block_seconds
                stats.add_block(current_block)
                write_session_summary(summary_path, session, device, stats, status="running")
                print(f"▶ 新ブロック: {current_block['block_id']}")

            try:
                frames = pipe.wait_for_frames(timeout_ms=WAIT_TIMEOUT_MS)
            except RsErr:
                continue

            rgb_vis = None
            ir_vis = None
            depth_vis = None

            cfrm = frames.get_color_frame()
            if cfrm:
                stats.increment("rgb", "acquired")
                rgb_img = np.asanyarray(cfrm.get_data()).copy()
                rgb_vis = rgb_img
                push_frame(
                    rgb_queue,
                    {
                        "block_id": current_block["block_id"],
                        "path": current_block["rgb_path"],
                        "frame": rgb_img,
                    },
                    stats,
                    "rgb",
                )

            ifrm = frames.get_infrared_frame()
            if ifrm:
                stats.increment("ir", "acquired")
                ir_img = np.asanyarray(ifrm.get_data(), dtype=np.uint8).copy()
                ir_vis = ir_img
                if should_skip_ir(rgb_queue, ir_queue):
                    stats.increment("ir", "skipped_policy")
                else:
                    push_frame(
                        ir_queue,
                        {
                            "block_id": current_block["block_id"],
                            "path": current_block["ir_path"],
                            "frame": ir_img,
                        },
                        stats,
                        "ir",
                    )

            dfrm = frames.get_depth_frame()
            if dfrm:
                depth_img = np.asanyarray(dfrm.get_data(), dtype=np.uint16).copy()
                depth_vis = depth_img
                stats.increment("depth", "acquired")
                if not depth_sampler.should_save():
                    stats.increment("depth", "skipped_sampling")
                elif should_skip_depth(rgb_queue, ir_queue, depth_queue):
                    stats.increment("depth", "skipped_policy")
                else:
                    push_frame(
                        depth_queue,
                        {
                            "block_id": current_block["block_id"],
                            "path": current_block["depth_path"],
                            "frame": depth_img,
                            "timestamp": dfrm.get_timestamp(),
                        },
                        stats,
                        "depth",
                    )

            if VISUALIZE and preview_frames(depth_vis, ir_vis, rgb_vis):
                stop_event.set()
                break

        if status == "running":
            status = "stopped"

    except KeyboardInterrupt:
        status = "stopped"
        print("\nユーザー停止")
    except Exception as exc:
        status = "error"
        error_message = str(exc)
        print(error_message, file=sys.stderr)
    finally:
        stop_event.set()

        if pipe is not None:
            try:
                pipe.stop()
            except Exception:
                pass

        for thread, q in worker_threads:
            put_stop_signal(q, thread)
        for thread, _ in worker_threads:
            thread.join(timeout=10)

        writer_errors: list[str] = []
        while not error_queue.empty():
            stream_name, exc = error_queue.get_nowait()
            writer_errors.append(f"{stream_name}: {exc}")
        if writer_errors and error_message is None:
            error_message = "; ".join(writer_errors)
            status = "error"

        if VISUALIZE:
            cv.destroyAllWindows()

        if "session" in locals() and "device" in locals():
            session["end_iso"] = utc_now_iso()
            write_session_summary(summary_path, session, device, stats, status=status, error_message=error_message)

        print("=== STABLE RECORD STOP ===")
        if error_message:
            sys.exit(error_message)


if __name__ == "__main__":
    main()
