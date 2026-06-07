"""Camera-teacher labeling: pose teacher + frame source + CSI, time-aligned to a
MM-Fi dataset (csi_amplitude.npy (N,56), keypoints.npy (N,51))."""
import argparse
import json
import os
import signal
import sys
import threading
import time
from collections import deque

import numpy as np

from .coco import COCO_NAMES, N_KP, MP_TO_COCO
from .csi import fuse_amplitude, SensingWS, N_SUB


# ── Frame sources ──────────────────────────────────────────────────────────
class FrameSource:
    """OpenCV V4L2/RTSP, or an HTTP JPEG snapshot endpoint (CSI/libcamera-safe)."""

    def __init__(self, spec):
        self.spec = str(spec)
        self.is_url = self.spec.startswith("http")
        self.cap = None
        if not self.is_url:
            import cv2
            self.cap = cv2.VideoCapture(int(self.spec) if self.spec.isdigit() else self.spec)

    def opened(self):
        return True if self.is_url else bool(self.cap and self.cap.isOpened())

    def read(self):
        if self.is_url:
            import urllib.request
            import cv2
            try:
                with urllib.request.urlopen(self.spec, timeout=4) as r:
                    buf = r.read()
                img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
                return img is not None, img
            except Exception:
                return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()


# ── Pose teachers ──────────────────────────────────────────────────────────
class YoloTeacher:
    name = "yolo"

    def __init__(self, min_conf=0.5, weights="yolov8n-pose.pt"):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def infer(self, frame):
        r = self.model.predict(frame, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None, 0.0
        xyn = r.keypoints.xyn[0].cpu().numpy()
        conf = r.keypoints.conf
        conf = conf[0].cpu().numpy() if conf is not None else np.ones(N_KP)
        kp = np.zeros((N_KP, 3), dtype=np.float32)
        kp[:, :2] = xyn[:N_KP]
        return kp, float(np.mean(conf))


class MediaPipeTeacher:
    name = "mediapipe"

    def __init__(self, min_conf=0.5):
        import mediapipe as mp
        self.pose = mp.solutions.pose.Pose(model_complexity=1,
                                           min_detection_confidence=min_conf,
                                           min_tracking_confidence=min_conf)

    def infer(self, frame):
        import cv2
        res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None, 0.0
        lm = res.pose_landmarks.landmark
        kp = np.zeros((N_KP, 3), dtype=np.float32)
        confs = []
        for i, mp_idx in enumerate(MP_TO_COCO):
            p = lm[mp_idx]
            kp[i] = (p.x, p.y, p.z)
            confs.append(getattr(p, "visibility", 1.0))
        return kp, float(np.mean(confs))


def make_teacher(backend, min_conf):
    return MediaPipeTeacher(min_conf) if backend == "mediapipe" else YoloTeacher(min_conf)


# ── CSI ring buffer (fed by SensingWS) ─────────────────────────────────────
class CsiBuffer:
    def __init__(self, ws_url, fuse="mean", maxlen=300):
        self.fuse = fuse
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.frames = 0
        self.node_count = 0
        self._ws = SensingWS(ws_url, self._on)
        self._t = None

    def _on(self, d):
        amp = fuse_amplitude(d.get("nodes"), mode=self.fuse)
        if amp is None:
            return
        ts = d.get("timestamp")
        ts = float(ts) if isinstance(ts, (int, float)) else time.time()
        with self.lock:
            self.buf.append((ts, amp))
            self.frames += 1
            self.node_count = len(d.get("nodes") or [])

    def start(self):
        self._t = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._t.start()

    def stop(self):
        self._ws.stop()

    def nearest(self, ts, tol):
        best, best_dt = None, tol
        with self.lock:
            for t, amp in reversed(self.buf):
                dt = abs(t - ts)
                if dt <= best_dt:
                    best_dt, best = dt, amp
                if ts - t > tol and best is not None:
                    break
        return best, best_dt


def collect(ws, camera, out, backend="yolo", fuse="mean", align_ms=60.0,
            min_pose_conf=0.5, max_samples=0, fps=8.0):
    import cv2  # noqa
    os.makedirs(out, exist_ok=True)
    tol = align_ms / 1000.0
    teacher = make_teacher(backend, min_pose_conf)
    csi = CsiBuffer(ws, fuse)
    csi.start()
    cap = FrameSource(camera)
    if not cap.opened():
        print(f"[teacher] ERROR: cannot open camera {camera}", file=sys.stderr)
        return 1

    csi_rows, kp_rows, errs = [], [], []
    frames = detections = 0
    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    print("[teacher] collecting… Ctrl-C to stop and save.")
    period = 1.0 / max(fps, 1.0)
    last = time.time()
    try:
        while not stop["v"]:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05); continue
            frames += 1
            kp, conf = teacher.infer(frame)
            ts = time.time()
            if kp is not None and conf >= min_pose_conf:
                detections += 1
                amp, dt = csi.nearest(ts, tol)
                if amp is not None:
                    csi_rows.append(amp); kp_rows.append(kp.reshape(-1)); errs.append(dt)
            if time.time() - last >= 1.0:
                with csi.lock:
                    cf, nc = csi.frames, csi.node_count
                ae = (sum(errs[-50:]) / len(errs[-50:]) * 1000) if errs else 0
                print(f"  frames={frames} pose={detections} paired={len(csi_rows)} "
                      f"csi_frames={cf} nodes={nc} align~{ae:.0f}ms", flush=True)
                last = time.time()
            if max_samples and len(csi_rows) >= max_samples:
                break
            d = period - (time.time() - t0)
            if d > 0:
                time.sleep(d)
    finally:
        cap.release(); csi.stop()

    n = len(csi_rows)
    if n == 0:
        print("[teacher] no paired samples — nothing saved.", file=sys.stderr)
        return 1
    np.save(os.path.join(out, "csi_amplitude.npy"), np.asarray(csi_rows, dtype=np.float32))
    np.save(os.path.join(out, "keypoints.npy"), np.asarray(kp_rows, dtype=np.float32))
    json.dump({"samples": n, "n_subcarriers": N_SUB, "backend": teacher.name, "fuse": fuse,
               "keypoint_format": "COCO17_xyz_normalized", "coco_order": COCO_NAMES,
               "align_ms_mean": (sum(errs) / n * 1000) if errs else 0,
               "frames_seen": frames, "poses_detected": detections, "dataset_type": "mmfi"},
              open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"[teacher] saved {n} samples -> {out}/ "
          f"(csi {N_SUB}, kp {N_KP*3}; mean align "
          f"{(sum(errs)/n*1000) if errs else 0:.0f}ms)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="WiFi-sense camera-teacher labeler")
    ap.add_argument("--ws", default="ws://localhost:13000/ws/sensing")
    ap.add_argument("--camera", default="http://localhost:8080/api/vision/snapshot?camera=0")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--backend", choices=["yolo", "mediapipe"], default="yolo")
    ap.add_argument("--fuse", choices=["mean", "node0"], default="mean")
    ap.add_argument("--align-ms", type=float, default=60.0)
    ap.add_argument("--min-pose-conf", type=float, default=0.5)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--fps", type=float, default=8.0)
    a = ap.parse_args()
    sys.exit(collect(a.ws, a.camera, a.out, a.backend, a.fuse, a.align_ms,
                     a.min_pose_conf, a.max_samples, a.fps))


if __name__ == "__main__":
    main()
