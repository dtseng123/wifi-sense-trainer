"""Headless stereo calibration for an SBS USB camera (e.g. the vr-passthrough
'3D USB Camera'). Show a checkerboard to BOTH eyes from varied angles/distances;
it auto-captures diverse valid pairs, runs stereo calibration + rectification, and
writes stereo_calib.json (per-eye K/D, R, T, baseline, rectification Q).

No GUI needed (works over SSH): just move the board around in front of the camera.

  wst-calibrate --camera /dev/video16 --cols 9 --rows 6 --square-mm 25 \
                --out dataset/stereo_calib.json
"""
import argparse
import json
import sys
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Stereo calibration for an SBS USB camera")
    ap.add_argument("--camera", default="/dev/video16")
    ap.add_argument("--width", type=int, default=1280, help="total SBS capture width")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--target", type=int, default=20, help="diverse pairs to collect")
    ap.add_argument("--min-move", type=float, default=30.0, help="min mean corner shift (px) to accept a new pair")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--swap-eyes", action="store_true")
    ap.add_argument("--out", default="stereo_calib.json")
    a = ap.parse_args()
    import cv2

    patt = (a.cols, a.rows)
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2)
    objp *= a.square_mm / 1000.0  # metres

    cap = cv2.VideoCapture(a.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        print(f"cannot open {a.camera}", file=sys.stderr); sys.exit(1)

    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def corners(gray):
        ok, c = cv2.findChessboardCorners(gray, patt, cb_flags)
        if not ok:
            return None
        return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)

    objpoints, lpts, rpts = [], [], []
    last = None
    size = None
    print(f"[calib] move the {a.cols}x{a.rows} board around in view of BOTH eyes "
          f"(target {a.target} pairs, {a.timeout:.0f}s)…", flush=True)
    t0 = time.time()
    seen = 0
    while len(objpoints) < a.target and time.time() - t0 < a.timeout:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.02); continue
        half = f.shape[1] // 2
        l, r = f[:, :half], f[:, half:half * 2]
        if a.swap_eyes:
            l, r = r, l
        gl = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        size = gl.shape[::-1]
        cl, cr = corners(gl), corners(gr)
        seen += 1
        if cl is None or cr is None:
            if seen % 30 == 0:
                print(f"  …searching (have {len(objpoints)}/{a.target})", flush=True)
            continue
        # accept only sufficiently different views (diversity)
        center = cl.mean(axis=0)
        if last is not None and np.linalg.norm(center - last) < a.min_move:
            continue
        last = center
        objpoints.append(objp.copy()); lpts.append(cl); rpts.append(cr)
        print(f"  captured pair {len(objpoints)}/{a.target}", flush=True)
    cap.release()

    if len(objpoints) < 6:
        print(f"[calib] only {len(objpoints)} pairs — need >=6. Improve lighting / board visibility.", file=sys.stderr)
        sys.exit(1)

    print(f"[calib] calibrating on {len(objpoints)} pairs…", flush=True)
    flags_mono = cv2.CALIB_RATIONAL_MODEL  # wide lens
    rl, K1, D1, _, _ = cv2.calibrateCamera(objpoints, lpts, size, None, None, flags=flags_mono)
    rr, K2, D2, _, _ = cv2.calibrateCamera(objpoints, rpts, size, None, None, flags=flags_mono)
    ret, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, lpts, rpts, K1, D1, K2, D2, size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
    baseline = float(np.linalg.norm(T))
    calib = {
        "image_size": list(size), "rms_left": rl, "rms_right": rr, "rms_stereo": ret,
        "K1": K1.tolist(), "D1": D1.tolist(), "K2": K2.tolist(), "D2": D2.tolist(),
        "R": R.tolist(), "T": T.tolist(), "R1": R1.tolist(), "R2": R2.tolist(),
        "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q.tolist(),
        "baseline_m": baseline, "fx": float(P1[0, 0]), "swap_eyes": a.swap_eyes,
    }
    json.dump(calib, open(a.out, "w"), indent=2)
    print(f"[calib] done: baseline={baseline*1000:.1f}mm  fx={calib['fx']:.1f}px  "
          f"stereo RMS={ret:.3f}px  -> {a.out}")
    print("[calib] (RMS < ~0.5px is good; > 1px means recapture with more varied, sharp views)")


if __name__ == "__main__":
    main()
