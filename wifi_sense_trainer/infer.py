"""Live self-inference: /ws/sensing CSI -> trained PoseHead -> a pluggable Sink.

The Sink decouples the model from any particular consumer. Built-in:
  HttpPostSink(url)  - POST {"persons":[...]} (e.g. optidex /api/ruview/pose)
  PrintSink()        - debug to stdout
  CallbackSink(fn)   - call your own function(persons)
"""
import argparse
import json
import sys
import time
import urllib.request

import numpy as np

from .coco import COCO_NAMES, N_KP
from .csi import fuse_amplitude, SensingWS
from .model import load_pose_head


# ── Sinks ──────────────────────────────────────────────────────────────────
class PrintSink:
    def __init__(self):
        self.n = 0

    def __call__(self, persons, src_ts=None):
        self.n += 1
        if self.n % 40 == 1:
            kp = persons[0]["keypoints"][0]
            print(f"[infer] {self.n} poses; nose=({kp['x']:.2f},{kp['y']:.2f})", flush=True)


class CallbackSink:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, persons, src_ts=None):
        self.fn(persons, src_ts)


class HttpPostSink:
    def __init__(self, url):
        self.url = url
        self.sent = 0

    def __call__(self, persons, src_ts=None):
        body = json.dumps({"persons": persons, "src_ts": src_ts, "source": "wifi-sense-trainer"}).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=2).read()
            self.sent += 1
            if self.sent % 40 == 1:
                print(f"[infer] emitted {self.sent} poses -> {self.url}", flush=True)
        except Exception as e:
            if self.sent % 50 == 0:
                print("[infer] post err", e, file=sys.stderr)


def _persons_from_output(out):
    kps = [{"name": COCO_NAMES[k], "x": float(out[k * 3]), "y": float(out[k * 3 + 1]),
            "z": float(out[k * 3 + 2]), "confidence": 0.8} for k in range(N_KP)]
    return [{"id": 1, "confidence": 0.85, "keypoints": kps}]


def run_infer(model_dir, ws, sink, max_hz=8.0, fuse="mean", gate_presence=False):
    import torch
    model, norm, arch = load_pose_head(model_dir)
    print(f"[infer] model {arch} loaded; ws={ws}", flush=True)
    st = {"last": 0.0}
    min_dt = 1.0 / max(max_hz, 0.5)

    def on_frame(d):
        now = time.time()
        if now - st["last"] < min_dt:
            return
        if gate_presence and not (d.get("classification") or {}).get("presence", True):
            return
        amp = fuse_amplitude(d.get("nodes"), mode=fuse)
        if amp is None:
            return
        x = norm.apply(amp)
        with torch.no_grad():
            out = model(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()
        st["last"] = now
        sink(_persons_from_output(out), d.get("timestamp"))

    SensingWS(ws, on_frame).run_forever()


def main():
    ap = argparse.ArgumentParser(description="Live WiFi-CSI pose self-inference sidecar")
    ap.add_argument("--ws", default="ws://localhost:13000/ws/sensing")
    ap.add_argument("--model-dir", default="dataset")
    ap.add_argument("--sink", default="http://localhost:8080/api/ruview/pose",
                    help="HTTP POST URL, or 'print'")
    ap.add_argument("--max-hz", type=float, default=8.0)
    ap.add_argument("--fuse", choices=["mean", "node0"], default="mean")
    ap.add_argument("--gate-presence", action="store_true")
    a = ap.parse_args()
    sink = PrintSink() if a.sink == "print" else HttpPostSink(a.sink)
    run_infer(a.model_dir, a.ws, sink, a.max_hz, a.fuse, a.gate_presence)


if __name__ == "__main__":
    main()
