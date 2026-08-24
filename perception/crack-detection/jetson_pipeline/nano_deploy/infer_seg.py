"""infer_seg.py - run a YOLO-seg TensorRT engine on the Jetson and save an overlay.

Full label-free deployment path for the original Jetson Nano (Python 3.6,
TensorRT 8.2, pycuda). Loads a ``.engine``, runs one image, decodes the
YOLO11-seg outputs (detection head + mask prototypes) by hand with numpy/cv2,
overlays the crack masks, and writes an annotated PNG plus a binary mask. Also
reports a per-stage timing breakdown so the fully-loaded FPS is honest.

Deliberately 3.6-safe: no ``from __future__`` import, no ``X | Y`` unions.

Usage:
    OPENBLAS_CORETYPE=ARMV8 python3 infer_seg.py \
        --engine models/yolo11n-seg_fp16.engine --image sample.png --out annotated.png
"""

# %% Imports
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 - creates the CUDA context

# %% Config
LETTERBOX_FILL = 114
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
MASK_THRESHOLD = 0.5
PROTO_SIZE     = 160
OVERLAY_COLOUR = (0, 0, 255)   # BGR red for crack pixels
OVERLAY_ALPHA  = 0.45


# %% Preprocess
def letterbox(image, size):
    """Resize preserving aspect ratio onto a square padded canvas.

    Args:
        image: HxWx3 BGR uint8 image.
        size: Target square side length.

    Returns:
        (canvas, scale, pad_x, pad_y).
    """
    height, width = image.shape[:2]
    scale = min(size / float(height), size / float(width))
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), LETTERBOX_FILL, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(image, size, dtype):
    """Letterbox, BGR->RGB, HWC->CHW, scale to [0,1].

    Args:
        image: HxWx3 BGR uint8 image.
        size: Square network input size.
        dtype: Target numpy dtype for the input binding.

    Returns:
        (tensor 1x3xSxS, scale, pad_x, pad_y).
    """
    canvas, scale, pad_x, pad_y = letterbox(image, size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None], dtype=dtype), scale, pad_x, pad_y


# %% Engine
def load_engine(path, logger):
    """Deserialise a TensorRT engine.

    Args:
        path: Path to the ``.engine`` file.
        logger: TensorRT logger.

    Returns:
        The deserialised engine.
    """
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
    if engine is None:
        raise RuntimeError("Failed to deserialize " + str(path))
    return engine


def allocate(engine, imgsz):
    """Allocate host/device buffers for every binding.

    Args:
        engine: The TensorRT engine.
        imgsz: Fallback input size for any dynamic dim.

    Returns:
        (inputs, outputs, bindings) - lists of buffer dicts and the ptr list.
    """
    inputs, outputs = [], []
    bindings = [0] * engine.num_bindings
    for i in range(engine.num_bindings):
        shape = tuple(engine.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(i))
        dims = [d if d > 0 else (1 if j == 0 else (3 if j == 1 else imgsz))
                for j, d in enumerate(shape)]
        entry = {
            "index": i, "name": engine.get_binding_name(i), "dims": dims, "dtype": dtype,
            "dev": cuda.mem_alloc(int(np.prod(dims)) * np.dtype(dtype).itemsize),
            "host": cuda.pagelocked_empty(int(np.prod(dims)), dtype),
        }
        bindings[i] = int(entry["dev"])
        (inputs if engine.binding_is_input(i) else outputs).append(entry)
    return inputs, outputs, bindings


# %% Postprocess
def sigmoid(x):
    """Numerically stable logistic sigmoid.

    Args:
        x: Input array.

    Returns:
        Elementwise sigmoid.
    """
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def find_output(outputs, channels):
    """Return the output buffer whose second dim matches ``channels``.

    Args:
        outputs: Output buffer dicts.
        channels: Channel count to match (37 for det head, 32 for protos).

    Returns:
        The matching buffer dict.
    """
    for out in outputs:
        if out["dims"][1] == channels:
            return out
    raise RuntimeError("No output with {} channels".format(channels))


def decode(outputs, scale, pad_x, pad_y, orig_w, orig_h, imgsz):
    """Decode YOLO-seg outputs into a single binary crack mask.

    Args:
        outputs: Output buffer dicts (already copied to host).
        scale: Letterbox scale factor.
        pad_x: Horizontal pad offset.
        pad_y: Vertical pad offset.
        orig_w: Native image width.
        orig_h: Native image height.
        imgsz: Network input size.

    Returns:
        (binary_mask HxW uint8 0/255, n_detections, boxes_xyxy_native).
    """
    det = np.asarray(find_output(outputs, 37)["host"]).reshape(37, -1).T   # (anchors, 37)
    proto_entry = find_output(outputs, 32)
    proto_h, proto_w = proto_entry["dims"][2], proto_entry["dims"][3]       # imgsz/4, scales with input
    proto = np.asarray(proto_entry["host"]).reshape(32, proto_h, proto_w)

    boxes_xywh = det[:, 0:4]
    scores = det[:, 4]
    coeffs = det[:, 5:37]

    keep = scores >= CONF_THRESHOLD
    boxes_xywh, scores, coeffs = boxes_xywh[keep], scores[keep], coeffs[keep]
    if boxes_xywh.shape[0] == 0:
        return np.zeros((orig_h, orig_w), np.uint8), 0, []

    # xywh (centre) -> xyxy in letterbox 640 space.
    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    nms_boxes = [[int(x[0]), int(x[1]), int(x[2] - x[0]), int(x[3] - x[1])] for x in xyxy]
    idx = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    if len(idx) == 0:
        return np.zeros((orig_h, orig_w), np.uint8), 0, []
    idx = np.array(idx).reshape(-1)

    # Mask prototypes: coeffs @ protos -> per-instance mask at 160x160.
    proto_flat = proto.reshape(32, -1)                     # (32, proto_h*proto_w)
    masks = sigmoid(coeffs[idx] @ proto_flat)              # (n, proto_h*proto_w)
    masks = masks.reshape(-1, proto_h, proto_w)

    union = np.zeros((imgsz, imgsz), np.float32)
    boxes_native = []
    for k, det_i in enumerate(idx):
        m = cv2.resize(masks[k], (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        bx = xyxy[det_i]
        x1, y1 = max(0, int(bx[0])), max(0, int(bx[1]))
        x2, y2 = min(imgsz, int(bx[2])), min(imgsz, int(bx[3]))
        crop = np.zeros((imgsz, imgsz), np.float32)
        crop[y1:y2, x1:x2] = m[y1:y2, x1:x2]               # confine mask to its box
        union = np.maximum(union, crop)
        boxes_native.append([
            int((bx[0] - pad_x) / scale), int((bx[1] - pad_y) / scale),
            int((bx[2] - pad_x) / scale), int((bx[3] - pad_y) / scale),
        ])

    # Un-letterbox the union mask to native resolution.
    binary_640 = (union >= MASK_THRESHOLD).astype(np.uint8) * 255
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    cropped = binary_640[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    native = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return native, int(len(idx)), boxes_native


def overlay(image, mask, boxes):
    """Composite the crack mask and boxes over the image.

    Args:
        image: HxWx3 BGR uint8 image.
        mask: HxW uint8 0/255 crack mask.
        boxes: List of [x1,y1,x2,y2] in native coords.

    Returns:
        Annotated HxWx3 BGR uint8 image.
    """
    out = image.copy()
    colour = np.zeros_like(out); colour[:] = OVERLAY_COLOUR
    sel = mask > 0
    out[sel] = (OVERLAY_ALPHA * colour[sel] + (1 - OVERLAY_ALPHA) * out[sel]).astype(np.uint8)
    for b in boxes:
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return out


# %% Main
def main(config):
    """Run inference and write the annotated overlay.

    Args:
        config: Runtime configuration.

    Returns:
        Process exit code.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(config["engine"], logger)
    context = engine.create_execution_context()
    inputs, outputs, bindings = allocate(engine, config["imgsz"])
    in_entry = inputs[0]
    imgsz = in_entry["dims"][-1]
    stream = cuda.Stream()

    image = cv2.imread(config["image"])
    if image is None:
        print("[error] could not read " + config["image"])
        return 1
    orig_h, orig_w = image.shape[:2]

    timings = {"pre": [], "infer": [], "post": []}
    mask = None; ndet = 0; boxes = []
    for r in range(config["runs"] + 1):
        t0 = time.perf_counter()
        tensor, scale, pad_x, pad_y = preprocess(image, imgsz, in_entry["dtype"])
        t1 = time.perf_counter()
        in_entry["host"][:] = tensor.ravel()
        cuda.memcpy_htod_async(in_entry["dev"], in_entry["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["dev"], stream)
        stream.synchronize()
        t2 = time.perf_counter()
        mask, ndet, boxes = decode(outputs, scale, pad_x, pad_y, orig_w, orig_h, imgsz)
        t3 = time.perf_counter()
        if r > 0:   # discard first as warmup
            timings["pre"].append(t1 - t0)
            timings["infer"].append(t2 - t1)
            timings["post"].append(t3 - t2)

    annotated = overlay(image, mask, boxes)
    cv2.imwrite(config["out"], annotated)
    cv2.imwrite(config["mask_out"], mask)

    pre = np.mean(timings["pre"]) * 1000
    inf = np.mean(timings["infer"]) * 1000
    post = np.mean(timings["post"]) * 1000
    total = pre + inf + post
    print("=== detections ===")
    print("  crack instances: {}  crack pixels: {} ({:.2f}% of image)".format(
        ndet, int((mask > 0).sum()), 100.0 * (mask > 0).mean()))
    print("=== timing (mean over {} runs, ms) ===".format(config["runs"]))
    print("  preprocess : {:7.2f}".format(pre))
    print("  inference  : {:7.2f}".format(inf))
    print("  postprocess: {:7.2f}".format(post))
    print("  TOTAL      : {:7.2f}  ->  {:.2f} FPS".format(total, 1000.0 / total))
    print("=== saved ===")
    print("  " + config["out"] + "  (annotated)")
    print("  " + config["mask_out"] + "  (binary mask)")
    return 0


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-seg TensorRT inference + overlay on the Jetson.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="annotated.png")
    parser.add_argument("--mask-out", dest="mask_out", default="mask.png")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    raise SystemExit(main(vars(args)))
