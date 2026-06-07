"""PoseHead: CSI amplitude vector -> 17 COCO keypoints (x, y, z) in [0, 1]."""
import torch.nn as nn


class PoseHead(nn.Module):
    def __init__(self, in_dim=56, hidden=256, out_dim=51, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim), nn.Sigmoid(),  # normalized keypoints
        )

    def forward(self, x):
        return self.net(x)


def load_pose_head(model_dir):
    """Load a trained PoseHead + its Normalizer from a model dir.
    Returns (model.eval(), normalizer, arch_dict)."""
    import os
    import torch
    from .csi import Normalizer
    ck = torch.load(os.path.join(model_dir, "pose_model.pt"), map_location="cpu")
    arch = ck["arch"]
    model = PoseHead(arch["in_dim"], arch["hidden"], arch["out_dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    norm = Normalizer.from_json(os.path.join(model_dir, "pose_norm.json"))
    return model, norm, arch
