#!/bin/bash
# INT8 for yolo11s with the mask-prototype branch pinned to FP16, then benchmark.
cd ~/crackseg
TRT=/usr/src/tensorrt/bin/trtexec
M=yolo11s-seg_crack_finetuned
echo "start $(date +%T)"
python3 build_int8.py models/$M.onnx engines/${M}_int8pin.engine 640 200 proto > logs/${M}_int8pin_build.log 2>&1
rc=$?
echo "build exit=$rc  $(date +%T)"
grep -E "pinned|OK:|FAILED|\[E\]" logs/${M}_int8pin_build.log | cut -c1-230 | head -6
if [ $rc -eq 0 ]; then
  $TRT --loadEngine=engines/${M}_int8pin.engine --noDataTransfers --useSpinWait --warmUp=1000 --duration=10 --avgRuns=100 > logs/${M}_int8pin.log 2>&1
  echo "benchmark exit=$?"
  grep -E "GPU Compute Time: min|Throughput:" logs/${M}_int8pin.log | sed 's/^\[[^]]*\] \[I\] //' | cut -c1-200
fi
echo "INT8PIN DONE $(date +%T)"