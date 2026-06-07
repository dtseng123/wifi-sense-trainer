"""PyTorch pose-head training (real backprop) on a MM-Fi-format dataset.

Trains CSI(56) -> keypoints(51) and saves pose_model.pt + pose_norm.json +
train_meta.json. Backprop is O(params)/batch, so this runs in seconds on a CPU
(unlike RuView's finite-difference Rust trainer, which is O(params*samples)).
"""
import argparse
import json
import os
import numpy as np

from .coco import COCO_NAMES
from .csi import Normalizer
from .model import PoseHead


def pck(pred, gt, thr):
    p = pred.reshape(-1, 17, 3)[:, :, :2]
    g = gt.reshape(-1, 17, 3)[:, :, :2]
    return float((np.linalg.norm(p - g, axis=2) < thr).mean())


def train(dataset="dataset", out="dataset", epochs=300, batch=64, lr=1e-3,
          hidden=256, patience=40, val_frac=0.2, seed=42):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)

    X = np.load(os.path.join(dataset, "csi_amplitude.npy")).astype(np.float32)
    Y = np.load(os.path.join(dataset, "keypoints.npy")).astype(np.float32)
    assert X.shape[0] == Y.shape[0], "csi/keypoints row mismatch"
    n, in_dim = X.shape
    out_dim = Y.shape[1]
    print(f"dataset: {n} samples csi={X.shape} kp={Y.shape}")

    norm = Normalizer.fit(X)
    Xn = norm.apply(X)
    idx = np.random.permutation(n)
    Xn, Y = Xn[idx], Y[idx]
    nv = max(1, int(n * val_frac))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xn[nv:], device=dev); Ytr = torch.tensor(Y[nv:], device=dev)
    Xva = torch.tensor(Xn[:nv], device=dev); Yva = torch.tensor(Y[:nv], device=dev)

    model = PoseHead(in_dim, hidden, out_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=12)
    lossf = nn.MSELoss()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam} device={dev} train={len(Xtr)} val={len(Xva)}")

    best, best_state, bad = float("inf"), None, 0
    ntr = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr, device=dev)
        tot = 0.0
        for i in range(0, ntr, batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), Ytr[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vp = model(Xva); vloss = lossf(vp, Yva).item()
        sched.step(vloss)
        if vloss < best - 1e-6:
            best = vloss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if ep % 20 == 0 or bad == 0:
            vpn, gvn = vp.cpu().numpy(), Yva.cpu().numpy()
            print(f"ep {ep:3d} train_mse {tot/max(ntr,1):.5f} val_mse {vloss:.5f} "
                  f"PCK@0.1 {pck(vpn,gvn,0.1):.3f} PCK@0.2 {pck(vpn,gvn,0.2):.3f}")
        if bad >= patience:
            print(f"early stop @ {ep}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        vpn, gvn = model(Xva).cpu().numpy(), Yva.cpu().numpy()
    final = {"val_mse": best, "pck@0.05": pck(vpn, gvn, 0.05),
             "pck@0.1": pck(vpn, gvn, 0.1), "pck@0.2": pck(vpn, gvn, 0.2)}
    print("BEST:", json.dumps(final))

    os.makedirs(out, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "arch": {"in_dim": in_dim, "hidden": hidden, "out_dim": out_dim}},
               os.path.join(out, "pose_model.pt"))
    norm.to_json(os.path.join(out, "pose_norm.json"),
                 {"in_dim": in_dim, "hidden": hidden, "out_dim": out_dim, "coco_order": COCO_NAMES})
    json.dump({"samples": n, "params": nparam, **final},
              open(os.path.join(out, "train_meta.json"), "w"), indent=2)
    print(f"saved -> {out}/pose_model.pt, pose_norm.json, train_meta.json")
    return final


def main():
    ap = argparse.ArgumentParser(description="Train WiFi-CSI pose head (backprop)")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--patience", type=int, default=40)
    a = ap.parse_args()
    train(a.dataset, a.out, a.epochs, a.batch, a.lr, a.hidden, a.patience)


if __name__ == "__main__":
    main()
