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
import os
import sys
import time
import numpy as np


def _solve_one(objpoints, lpts, rpts, size, focal_guess, extra_flag):
    """One full mono+stereo+rectify solve with a given distortion model flag.
    Returns (calib_dict, stereo_rms, baseline_m) or None if it diverges."""
    import cv2
    w, h = size
    K0 = np.array([[focal_guess, 0, w / 2.0], [0, focal_guess, h / 2.0], [0, 0, 1]], np.float64)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    base = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT
            + cv2.CALIB_FIX_ASPECT_RATIO + extra_flag)
    try:
        rl, K1, D1, _, _ = cv2.calibrateCamera(objpoints, lpts, size, K0.copy(), None, flags=base)
        rr, K2, D2, _, _ = cv2.calibrateCamera(objpoints, rpts, size, K0.copy(), None, flags=base)
        ret, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
            objpoints, lpts, rpts, K1, D1, K2, D2, size,
            flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
    except cv2.error:
        return None
    baseline = float(np.linalg.norm(T))
    calib = {
        "image_size": list(size), "rms_left": float(rl), "rms_right": float(rr),
        "rms_stereo": float(ret), "K1": K1.tolist(), "D1": D1.tolist(),
        "K2": K2.tolist(), "D2": D2.tolist(), "R": R.tolist(), "T": T.tolist(),
        "R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(), "P2": P2.tolist(),
        "Q": Q.tolist(), "baseline_m": baseline, "fx": float(P1[0, 0]),
    }
    return calib, float(ret), baseline


def solve_and_write(objpoints, lpts, rpts, size, focal_guess, swap_eyes, out):
    """Try several distortion models (wide-lens RATIONAL first, then THIN_PRISM,
    then standard 5-coeff) and keep the most physical, lowest-RMS result. A 125°
    lens needs the richer models; standard alone leaves ~2-3px residual."""
    import cv2
    objpoints = [np.asarray(o, np.float32) for o in objpoints]
    lpts = [np.asarray(p, np.float32) for p in lpts]
    rpts = [np.asarray(p, np.float32) for p in rpts]
    models = [("rational", cv2.CALIB_RATIONAL_MODEL),
              ("rational+thinprism", cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL),
              ("standard", 0)]
    best = None
    for name, flag in models:
        res = _solve_one(objpoints, lpts, rpts, size, focal_guess, flag)
        if res is None:
            print(f"[calib]   model {name}: diverged"); continue
        calib, ret, baseline = res
        # "usable" = physically-plausible geometry. RMS < 1px is ideal, but up to
        # ~2px is workable for relative pose depth (residual = board non-flatness).
        physical = ret < 2.2 and 0.03 < baseline < 0.20 and 100 < calib["fx"] < 1500
        print(f"[calib]   model {name}: RMS={ret:.3f}px baseline={baseline*1000:.1f}mm "
              f"fx={calib['fx']:.0f}px {'OK' if physical else 'non-physical'}")
        score = ret if physical else ret + 1e6
        if best is None or score < best[0]:
            best = (score, name, calib, physical)
    if best is None:
        print("[calib] all models diverged.", file=sys.stderr); return 2
    _, name, calib, physical = best
    calib["swap_eyes"] = swap_eyes
    calib["distortion_model"] = name
    json.dump(calib, open(out, "w"), indent=2)
    status = "OK" if physical else "SUSPECT — DO NOT USE"
    print(f"[calib] best=[{name}] [{status}]: baseline={calib['baseline_m']*1000:.1f}mm "
          f"fx={calib['fx']:.1f}px stereo RMS={calib['rms_stereo']:.3f}px -> {out}")
    if not physical:
        print("[calib] Still non-physical. Most likely the board isn't rigid/flat. "
              "Tape the print to stiff card/glass, recapture across the whole frame "
              "with strong tilts. (Raw corners saved; re-solve with --resolve-from.)",
              file=sys.stderr)
        return 2
    print("[calib] (RMS < ~0.5px excellent; < 1px fine)")
    return 0


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
    ap.add_argument("--min-sharp", type=float, default=60.0, help="min board-ROI Laplacian variance (reject blur)")
    ap.add_argument("--focal-guess", type=float, default=400.0, help="initial fx/fy seed (px) — regularizes the solve")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--swap-eyes", action="store_true")
    ap.add_argument("--rotate-180", action="store_true", help="camera mounted upside down")
    ap.add_argument("--out", default="stereo_calib.json")
    ap.add_argument("--resolve-from", default=None,
                    help="re-solve from a saved *_corners.npz (no camera/capture needed)")
    a = ap.parse_args()

    if a.resolve_from:
        d = np.load(a.resolve_from)
        op = [x for x in d["objpoints"]]; lp = [x for x in d["lpts"]]; rp = [x for x in d["rpts"]]
        size = tuple(int(v) for v in d["size"])
        print(f"[calib] re-solving from {a.resolve_from}: {len(op)} pairs, size={size}")
        sys.exit(solve_and_write(op, lp, rp, size, a.focal_guess, a.swap_eyes, a.out))
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

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    have_sb = hasattr(cv2, "findChessboardCornersSB")

    def corners(gray):
        # SB detector: robust to blur AND gives a deterministic corner ordering,
        # so the SAME physical corner is index 0 in both eyes (no 180° mismatch).
        if have_sb:
            ok, c = cv2.findChessboardCornersSB(
                gray, patt, cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ACCURACY)
            if ok:
                return c
        ok, c = cv2.findChessboardCorners(gray, patt, cb_flags)
        if not ok:
            return None
        return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)

    def canon(c):
        # The board has only a 180° detection ambiguity (a 9x6 pattern is found
        # upright or rotated 180°, never 90°). Lock orientation per-eye using ROW
        # geometry (mean y of the first vs last grid row) — far more stable than a
        # single-corner test. After this both eyes are "top row up" row-major and
        # map onto one FIXED objp.
        g = c.reshape(a.rows, a.cols, 2)
        if g[0, :, 1].mean() > g[-1, :, 1].mean():
            g = g[::-1, ::-1]          # 180° (the only correspondence-preserving relabel)
        return g.reshape(-1, 1, 2)

    def lock_right_to_left(cl, cr):
        # Belt-and-suspenders cross-eye lock: in a ~horizontal rig matched corners
        # share rows (small |Δy|). Pick cr vs its 180° twin by min vertical error.
        yl = cl.reshape(-1, 2)[:, 1]
        e0 = np.abs(yl - cr.reshape(-1, 2)[:, 1]).mean()
        cr2 = cr.reshape(a.rows, a.cols, 2)[::-1, ::-1].reshape(-1, 1, 2)
        e1 = np.abs(yl - cr2.reshape(-1, 2)[:, 1]).mean()
        return cr2 if e1 < e0 else cr

    objpoints, lpts, rpts = [], [], []
    last = None          # last ACCEPTED pose (diversity)
    prev = None          # previous frame's corners (stillness)
    size = None
    miss = {"l": 0, "r": 0, "both": 0, "move": 0, "dup": 0}
    print(f"[calib] show the {a.cols}x{a.rows} board to BOTH eyes — move to a new "
          f"pose, then HOLD STILL for a beat; repeat (target {a.target}, {a.timeout:.0f}s)…",
          flush=True)
    t0 = time.time()
    seen = 0
    while len(objpoints) < a.target and time.time() - t0 < a.timeout:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.02); continue
        if a.rotate_180:
            f = cv2.rotate(f, cv2.ROTATE_180)
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
            prev = None
            if cl is None and cr is None:
                miss["both"] += 1
            elif cr is None:
                miss["r"] += 1       # left sees it, right doesn't -> board too far RIGHT
            else:
                miss["l"] += 1       # right sees it, left doesn't -> board too far LEFT
            if seen % 25 == 0:
                hint = ("board not seen — center it" if miss["both"] >= max(miss["l"], miss["r"])
                        else "shift board LEFT (right eye can't see it)" if miss["r"] > miss["l"]
                        else "shift board RIGHT (left eye can't see it)")
                print(f"  …{len(objpoints)}/{a.target}: {hint}", flush=True)
                miss = {k: 0 for k in miss}
            continue
        cl_flat = cl.reshape(-1, 2)
        # stillness gate: board must be steady vs the previous frame (sharp capture)
        motion = (np.linalg.norm(cl_flat - prev, axis=1).mean()
                  if prev is not None and prev.shape == cl_flat.shape else 1e9)
        prev = cl_flat
        if motion > 3.0:
            miss["move"] += 1
            if seen % 25 == 0:
                print(f"  …{len(objpoints)}/{a.target}: both eyes see it — HOLD STILL", flush=True)
            continue
        # sharpness gate: reject motion-blurred frames (Laplacian variance on the board ROI)
        x0, y0 = cl_flat.min(0); x1, y1 = cl_flat.max(0)
        roi = gl[max(int(y0), 0):int(y1) + 1, max(int(x0), 0):int(x1) + 1]
        sharp = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size else 0.0
        if sharp < a.min_sharp:
            if seen % 25 == 0:
                print(f"  …{len(objpoints)}/{a.target}: too blurry ({sharp:.0f}) — hold steadier / more light", flush=True)
            continue
        # tilt signature: skew of the board quad (breaks the focal/distance ambiguity)
        g = cl_flat.reshape(a.rows, a.cols, 2)
        d1 = np.linalg.norm(g[0, 0] - g[-1, -1]); d2 = np.linalg.norm(g[0, -1] - g[-1, 0])
        tilt = float(min(d1, d2) / max(d1, d2))   # 1.0 = fronto-parallel, lower = tilted
        # diversity gate: pose must differ in POSITION or TILT from the last accepted
        center = cl_flat.mean(axis=0)
        if last is not None and np.linalg.norm(center - last[:2]) < a.min_move and abs(tilt - last[2]) < 0.06:
            if seen % 25 == 0:
                print(f"  …{len(objpoints)}/{a.target}: TILT the board more (angle it), or move it", flush=True)
            continue
        last = np.array([center[0], center[1], tilt])
        cl, cr = canon(cl), lock_right_to_left(canon(cl), canon(cr))
        objpoints.append(objp.copy()); lpts.append(cl); rpts.append(cr)
        print(f"  captured pair {len(objpoints)}/{a.target}  (tilt={tilt:.2f})", flush=True)
    cap.release()

    if len(objpoints) < 6:
        print(f"[calib] only {len(objpoints)} pairs — need >=6. Improve lighting / board visibility.", file=sys.stderr)
        sys.exit(1)

    # Save the raw correspondences so the (cheap) solve can be re-run offline with
    # different distortion models WITHOUT recapturing — `wst-calibrate --resolve-from`.
    npz = os.path.splitext(a.out)[0] + "_corners.npz"
    np.savez(npz, objpoints=np.asarray(objpoints), lpts=np.asarray(lpts),
             rpts=np.asarray(rpts), size=np.asarray(size))
    print(f"[calib] saved {len(objpoints)} corner pairs -> {npz} (re-solve with --resolve-from)")
    sys.exit(solve_and_write(objpoints, lpts, rpts, tuple(size), a.focal_guess, a.swap_eyes, a.out))


if __name__ == "__main__":
    main()
