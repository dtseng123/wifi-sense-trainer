# wifi-sense-trainer

Train and deploy a **real WiFi-CSI human-pose head** on top of a [RuView](https://github.com/ruvnet/RuView)
CSI aggregator — camera-teacher auto-labeling, PyTorch training (real backprop),
and a live self-inference sidecar. Camera-free at inference.

## Why this exists
RuView ships no learned pose weights; its Rust `--train` uses finite-difference
gradients (≈`2·N_params` loss evals per batch) that never finish for the ~30k-param
pose transformer; and its live server pose is a hardwired geometric heuristic that
ignores any loaded model. This package sidesteps all three: it trains a small head
with **proper backprop** (seconds on a CPU) and runs inference itself, feeding any
consumer via a pluggable sink.

```
 camera ─► pose teacher (YOLOv8 / MediaPipe) ─► 17 COCO keypoints (t)
                                                   │ time-align
 ESP32 nodes ─► aggregator /ws/sensing ─► fused CSI amplitude[56] (t)
                                                   ▼
   dataset (MM-Fi): csi_amplitude.npy (N,56) + keypoints.npy (N,51)
                                                   ▼
            wst-train  ─►  pose_model.pt + pose_norm.json
                                                   ▼
            wst-infer  ─►  Sink (HTTP POST / callback / print)
```

## Prerequisites — a running RuView aggregator
This package trains *on top of* a [RuView](https://github.com/ruvnet/RuView) CSI
aggregator that exposes `/ws/sensing`. You need:

1. **ESP32-S3 CSI nodes** flashed + provisioned to stream CSI to the aggregator
   (see RuView's `firmware/esp32-csi-node` docs). They must emit **per-node
   `amplitude`** in the frames — i.e. a CSI-feature edge tier, not just on-device
   vitals.
2. **The aggregator** (prebuilt multi-arch image):
   ```bash
   docker run -e CSI_SOURCE=esp32 -p 13000:3000 -p 5005:5005/udp ruvnet/wifi-densepose:latest
   ```
   Confirm a `/ws/sensing` frame contains `nodes[].amplitude` (3+ nodes recommended).
3. **A camera** that sees the sensing area — a USB webcam index, an RTSP URL, or any
   HTTP JPEG snapshot endpoint (`--camera "http://host/snapshot.jpg"`).

> Why not just load a model into RuView? Its live server pose is a hardwired
> geometric heuristic that ignores trained weights, and its Rust trainer uses
> finite-difference gradients that don't finish for a real model. So you train
> here and **consume the poses yourself** via a Sink (below).

## Install
```bash
python3 -m venv --system-site-packages .venv     # inherit system numpy/cv2/torch/ultralytics
. .venv/bin/activate
pip install -e .            # core; add [teacher] / [torch] extras if not on the system
```
(MediaPipe has no wheel for Python 3.13 — YOLOv8-pose is the default teacher.)

## Use
```bash
# 1) collect (person moves in view of the camera, near the CSI rig)
wst-collect --ws ws://HOST:13000/ws/sensing \
            --camera "http://HOST:8080/api/vision/snapshot?camera=0" --out dataset

# 2) train (real backprop; runs on a CPU)
wst-train --dataset dataset --out dataset

# 3) live inference -> your consumer
wst-infer --model-dir dataset --ws ws://HOST:13000/ws/sensing \
          --sink http://HOST:8080/api/ruview/pose       # or --sink print
```

## Design / reuse
- **`csi`** — `fuse_amplitude`, `Normalizer`, `SensingWS` (reconnecting `/ws/sensing` client).
- **`coco`** — COCO-17 names, BlazePose→COCO map, skeleton edges.
- **`model`** — `PoseHead` + `load_pose_head`.
- **`teacher` / `train` / `infer`** — pipeline stages, each with a CLI entrypoint.
- **Sinks** decouple the model from any framework. The optidex panel is just one
  `HttpPostSink` consumer; swap in `CallbackSink(fn)` to embed elsewhere.

## Honest limits (not compute)
Accuracy is gated by **rig geometry** (rigid + spread nodes), **dataset size/variety**,
and the inherent coarseness of WiFi-CSI pose — not by GPU. A compact, non-fixed rig
gives coarse pose and must be retrained when re-placed; rigidize for reuse.

MIT licensed.
