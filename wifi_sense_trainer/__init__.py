"""wifi-sense-trainer: train & deploy a WiFi-CSI pose head on a RuView aggregator.

Modules:
  coco    - COCO-17 keypoint names, BlazePose->COCO map, skeleton edges
  csi     - CSI amplitude fusion, z-score Normalizer, /ws/sensing client
  model   - PoseHead (CSI -> 17 keypoints) MLP
  teacher - camera pose teacher + frame sources -> MM-Fi dataset
  train   - PyTorch training loop (real backprop) + metrics
  infer   - live inference sidecar with a pluggable Sink
"""
__version__ = "0.1.0"

from . import coco, csi  # lightweight, always importable

__all__ = ["coco", "csi", "__version__"]
