"""
eval_seg.py - mask IoU/Dice of YOLO11-seg TensorRT engines vs ground truth, plus engine-vs-engine agreement.

Usage:
    python3 eval_seg.py IMG_DIR LABEL_DIR ENGINE [ENGINE ...]
"""

# %% Imports
import ctypes
import glob
import json
import os
import sys

import cv2
import numpy as np
import tensorrt as trt


# %% Config
CONF_THRES = float(os.environ.get("CONF", 0.25))
IOU_THRES  = 0.7
MASK_THRES = 0.5
IMGSZ      = 640

cudart = ctypes.CDLL("libcudart.so.12")
LOGGER = trt.Logger(trt.Logger.ERROR)
NP_DT  = {trt.float32: np.float32, trt.float16: np.float16}


# %% CUDA helpers
def cu(ret: int) -> None:
    if ret != 0:
        raise RuntimeError(f"cuda error {ret}")


def cu_malloc(n: int) -> int:
    p = ctypes.c_void_p()
    cu(cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)))
    return p.value


# %% Engine wrapper
class Engine:
    def __init__(self, path: str) -> None:
        rt          = trt.Runtime(LOGGER)
        self.engine = rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx    = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        cu(cudart.cudaStreamCreate(ctypes.byref(self.stream)))
        self.inp, self.outs, self.bufs = None, [], {}
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dt    = NP_DT[self.engine.get_tensor_dtype(name)]
            nbyte = int(np.prod(shape)) * np.dtype(dt).itemsize
            dev   = cu_malloc(nbyte)
            self.ctx.set_tensor_address(name, dev)
            host  = np.empty(shape, dt)
            self.bufs[name] = (dev, host, nbyte)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inp = name
            else:
                self.outs.append(name)

    def __call__(self, x: np.ndarray) -> list:
        dev, host, nbyte = self.bufs[self.inp]
        host[...] = x.astype(host.dtype)
        cu(cudart.cudaMemcpyAsync(ctypes.c_void_p(dev), host.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(nbyte), 1, self.stream))
        self.ctx.execute_async_v3(self.stream.value)
        res = []
        for n in self.outs:
            d, h, nb = self.bufs[n]
            cu(cudart.cudaMemcpyAsync(h.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(d), ctypes.c_size_t(nb), 2, self.stream))
            res.append(h)
        cu(cudart.cudaStreamSynchronize(self.stream))
        return [r.astype(np.float32) for r in res]


# %% Pre/post-processing
def letterbox(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    r    = min(IMGSZ / h, IMGSZ / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), (top, left, nh, nw)


def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a))


def decode_yolo11(outs: list, pad: tuple, hw: tuple) -> np.ndarray:
    pred   = outs[0][0]
    protos = outs[1][0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                                   # [8400, 37]
    scores = pred[:, 4]
    keep   = scores > CONF_THRES
    full   = np.zeros(hw, np.uint8)
    if not keep.any():
        return full
    p      = pred[keep]
    boxes  = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 2], p[:, 3]], 1)
    idx    = cv2.dnn.NMSBoxes(boxes.tolist(), p[:, 4].tolist(), CONF_THRES, IOU_THRES)
    idx    = np.array(idx).reshape(-1)
    top, left, nh, nw = pad
    nm, mh, mw = protos.shape
    pm     = sigmoid(p[idx, 5:5 + nm] @ protos.reshape(nm, -1)).reshape(-1, mh, mw)
    for k, i in enumerate(idx):
        x0, y0, bw, bh = boxes[i]
        m  = cv2.resize(pm[k], (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        cm = np.zeros_like(m)
        x1, y1, x2, y2 = int(max(x0, 0)), int(max(y0, 0)), int(min(x0 + bw, IMGSZ)), int(min(y0 + bh, IMGSZ))
        cm[y1:y2, x1:x2] = m[y1:y2, x1:x2]
        cm = cm[top:top + nh, left:left + nw]
        cm = cv2.resize(cm, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
        full |= (cm > MASK_THRES).astype(np.uint8)
    return full


def load_gt(path: str, hw: tuple) -> np.ndarray:
    m = np.zeros(hw, np.uint8)
    for line in open(path).read().strip().splitlines():
        v = np.array(line.split()[1:], np.float32).reshape(-1, 2)
        pts = (v * [hw[1], hw[0]]).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def iou_dice(a: np.ndarray, b: np.ndarray) -> tuple:
    inter, ua, ub = int((a & b).sum()), int(a.sum()), int(b.sum())
    union = ua + ub - inter
    iou   = 1.0 if union == 0 else inter / union
    dice  = 1.0 if ua + ub == 0 else 2 * inter / (ua + ub)
    return iou, dice


# %% Main
def main(img_dir: str, lab_dir: str, engines: list) -> None:
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    names = [os.path.basename(e) for e in engines]
    preds = {n: [] for n in names}
    gts, gt_frac = [], []
    for e, n in zip(engines, names):
        eng = Engine(e)
        for f in files:
            img = cv2.imread(f)
            x, pad = letterbox(img)
            preds[n].append(decode_yolo11(eng(x), pad, img.shape[:2]))
        print(f"done {n}", flush=True)
    for f in files:
        img = cv2.imread(f)
        g   = load_gt(os.path.join(lab_dir, os.path.splitext(os.path.basename(f))[0] + ".txt"), img.shape[:2])
        gts.append(g)
        gt_frac.append(g.mean())
    gt_frac = np.array(gt_frac)
    q1, q2  = np.quantile(gt_frac, [1 / 3, 2 / 3])
    bucket  = np.where(gt_frac <= q1, 0, np.where(gt_frac <= q2, 1, 2))
    out = {}
    print(f"\nconf={CONF_THRES}  n={len(files)}  thin-bucket GT area<={q1:.4f}, mid<={q2:.4f}")
    print(f"{'engine':46s} {'IoU':>6s} {'Dice':>6s} {'thin':>6s} {'mid':>6s} {'thick':>6s} {'miss':>5s}")
    for n in names:
        r = np.array([iou_dice(p, g) for p, g in zip(preds[n], gts)])
        miss = sum(1 for p in preds[n] if p.sum() == 0)
        b = [r[bucket == k, 0].mean() for k in range(3)]
        print(f"{n:46s} {r[:,0].mean():6.4f} {r[:,1].mean():6.4f} {b[0]:6.4f} {b[1]:6.4f} {b[2]:6.4f} {miss:5d}")
        out[n] = dict(iou=float(r[:, 0].mean()), dice=float(r[:, 1].mean()), thin=float(b[0]), mid=float(b[1]), thick=float(b[2]), missed=miss,
                      per_image_iou=r[:, 0].tolist())
    if len(names) > 1:
        print("\nagreement with first engine (mask IoU between engines):")
        for n in names[1:]:
            a = np.array([iou_dice(p, q)[0] for p, q in zip(preds[names[0]], preds[n])])
            d = np.array(out[n]["per_image_iou"]) - np.array(out[names[0]]["per_image_iou"])
            print(f"  {n}: agreement {a.mean():.4f}  | per-image IoU delta mean {d.mean():+.4f}, worse on {(d < -0.02).sum()} imgs, better on {(d > 0.02).sum()} imgs, worst drop {d.min():+.3f}")
    json.dump(out, open("eval_result.json", "w"))


# %% Entry point
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
