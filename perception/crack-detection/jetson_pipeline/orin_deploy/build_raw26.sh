#!/bin/bash
cd ~/crackseg
TRT=/usr/src/tensorrt/bin/trtexec
BENCH="--noDataTransfers --useSpinWait --warmUp=1000 --duration=10 --avgRuns=100"
echo "power: $(nvpmodel -q 2>&1 | head -1)  start $(date +%T)"
for m in yolo26n-seg_crack_raw yolo26s-seg_crack_raw; do
  echo "=== $m FP16 $(date +%T) ==="
  $TRT --onnx=models/$m.onnx --fp16 --saveEngine=engines/${m}_fp16.engine --memPoolSize=workspace:2048 $BENCH > logs/${m}_fp16.log 2>&1
  echo "fp16 exit=$?"
  echo "=== $m INT8(pin proto) build $(date +%T) ==="
  python3 build_int8.py models/$m.onnx engines/${m}_int8pin.engine 640 200 proto > logs/${m}_int8pin_build.log 2>&1
  rc=$?; echo "build exit=$rc"; tail -n 2 logs/${m}_int8pin_build.log | cut -c1-200
  if [ $rc -eq 0 ]; then $TRT --loadEngine=engines/${m}_int8pin.engine $BENCH > logs/${m}_int8pin.log 2>&1; fi
  echo "--- $m done $(date +%T) ---"
done
echo ALL_DONE