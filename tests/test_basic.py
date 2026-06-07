"""Lightweight tests (no hardware / no torch required for most)."""
import numpy as np
from wifi_sense_trainer import coco
from wifi_sense_trainer.csi import fuse_amplitude, Normalizer, N_SUB


def test_coco_shapes():
    assert len(coco.COCO_NAMES) == 17
    assert len(coco.MP_TO_COCO) == 17
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in coco.SKELETON_EDGES)


def test_fuse_resamples_and_means():
    nodes = [{"amplitude": list(np.ones(30))}, {"amplitude": list(np.ones(56) * 3)}]
    amp = fuse_amplitude(nodes)
    assert amp.shape == (N_SUB,) and amp.dtype == np.float32
    assert abs(float(amp.mean()) - 2.0) < 0.1  # mean of 1s and 3s
    assert fuse_amplitude([]) is None
    assert fuse_amplitude(None) is None


def test_normalizer_roundtrip():
    X = np.random.RandomState(0).randn(100, N_SUB).astype(np.float32) * 5 + 2
    nz = Normalizer.fit(X)
    Xn = nz.apply(X)
    assert abs(float(Xn.mean())) < 0.1 and abs(float(Xn.std()) - 1.0) < 0.1


def test_pose_head_forward():
    import torch  # skip if torch absent
    from wifi_sense_trainer.model import PoseHead
    m = PoseHead(56, 64, 51).eval()
    with torch.no_grad():
        y = m(torch.zeros(4, 56))
    assert y.shape == (4, 51)
    assert (y >= 0).all() and (y <= 1).all()  # sigmoid range
