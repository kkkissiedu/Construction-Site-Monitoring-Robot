"""
phone_probe.py - headless link test for the Record3DStream iPhone server: frame rate, jitter, depth quality, pose drift.

Usage:
    python3 phone_probe.py 192.168.100.49 --seconds 30
"""

# %% Imports
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/phone_ws/Record3DStream/sdk"))
from sdk import IPhoneSensorClient


# %% Config
PORT        = 8888
PATCH_HALF  = 10          # centre patch is (2*PATCH_HALF)^2 depth pixels
OUT_DIR     = os.path.expanduser("~/phone_ws/out")


# %% Helpers
def pct(a: list, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


# %% Main
def main(host: str, seconds: float) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    client = IPhoneSensorClient(host, PORT)
    if not client.start():
        print("CONNECT FAILED")
        return
    arrivals, ts, ids, centre, conf_hi, poses = [], [], [], [], [], []
    first, saved = None, False
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            f = client.wait_for_frame(timeout=2.0)
            if f is None:
                continue
            arrivals.append(time.monotonic())
            ts.append(f.timestamp)
            ids.append(f.frame_id)
            h, w = f.depth.shape
            patch = f.depth[h // 2 - PATCH_HALF:h // 2 + PATCH_HALF, w // 2 - PATCH_HALF:w // 2 + PATCH_HALF]
            valid = patch[(patch > 0) & np.isfinite(patch)]
            centre.append(float(np.median(valid)) if valid.size else float("nan"))
            if f.confidence is not None:
                conf_hi.append(float((f.confidence == 2).mean()))
            poses.append(f.transform[:3, 3].copy())
            if first is None:
                first = f
            if not saved and len(arrivals) == 30:
                cv2.imwrite(os.path.join(OUT_DIR, "color.jpg"), f.color)
                d = np.clip(f.depth / 5.0, 0, 1)
                cv2.imwrite(os.path.join(OUT_DIR, "depth.png"), cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
                saved = True
        stats = client.get_stats()
    finally:
        client.stop()
    if len(arrivals) < 3:
        print(f"only {len(arrivals)} frames received")
        return
    dt  = np.diff(arrivals) * 1000.0
    dts = np.diff(ts) * 1000.0
    poses = np.array(poses)
    res = dict(
        frames        = len(arrivals),
        seconds       = round(arrivals[-1] - arrivals[0], 2),
        fps           = round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 2),
        dropped_ids   = int(ids[-1] - ids[0] + 1 - len(ids)),
        arrival_ms    = dict(mean=round(float(dt.mean()), 1), p50=round(pct(dt, 50), 1), p95=round(pct(dt, 95), 1), p99=round(pct(dt, 99), 1), max=round(float(dt.max()), 1)),
        device_dt_ms  = dict(mean=round(float(dts.mean()), 1), std=round(float(dts.std()), 1)),
        color_shape   = list(first.color.shape),
        depth_shape   = list(first.depth.shape),
        fx_fy         = [round(float(first.intrinsics.fx), 1), round(float(first.intrinsics.fy), 1)],
        centre_depth_m= dict(median=round(float(np.nanmedian(centre)), 3), std=round(float(np.nanstd(centre)), 4)),
        conf_high_frac= round(float(np.mean(conf_hi)), 3) if conf_hi else None,
        pose_span_m   = [round(float(v), 4) for v in (poses.max(0) - poses.min(0))],
        sdk_stats     = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()},
    )
    print(json.dumps(res, indent=2, default=str))
    json.dump(res, open(os.path.join(OUT_DIR, "probe_result.json"), "w"), indent=2, default=str)


# %% Entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("--seconds", type=float, default=30)
    a = ap.parse_args()
    main(a.host, a.seconds)
