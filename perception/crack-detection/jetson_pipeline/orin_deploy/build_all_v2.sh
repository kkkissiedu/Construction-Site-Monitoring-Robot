#!/bin/bash
# New order: the 640 px models first (FP16 then INT8 each), then 512 px, then 416 px. Skips FP16 work already done.
cd ~/crackseg
TRT=/usr/src/tensorrt/bin/trtexec
MODELS="yolo11s-seg_crack_finetuned yolo26s-seg_crack_finetuned yolo26n-seg_crack_finetuned yolo11n-seg_crack_512 yolo11n-seg_crack_416"
declare -A SZ=( [yolo11n-seg_crack_416]=416 [yolo11n-seg_crack_512]=512 [yolo11s-seg_crack_finetuned]=640 [yolo26n-seg_crack_finetuned]=640 [yolo26s-seg_crack_finetuned]=640 )
BENCH="--noDataTransfers --useSpinWait --warmUp=1000 --duration=10 --avgRuns=100"
echo "power mode: $(nvpmodel -q 2>&1 | head -1)   start: $(date +%T)"
for m in $MODELS; do
  if [ -f logs/${m}_fp16.log ] && grep -q "GPU Compute Time: min" logs/${m}_fp16.log; then
    echo "=== $m FP16: already done, skipping ==="
  else
    echo "=== $m FP16 build+bench  $(date +%T) ==="
    $TRT --onnx=models/$m.onnx --fp16 --saveEngine=engines/${m}_fp16.engine --memPoolSize=workspace:2048 $BENCH > logs/${m}_fp16.log 2>&1
    echo "exit=$?"
  fi
  echo "=== $m INT8 build  $(date +%T) ==="
  python3 build_int8.py models/$m.onnx engines/${m}_int8.engine ${SZ[$m]} 200 > logs/${m}_int8_build.log 2>&1
  rc=$?; echo "build exit=$rc"; tail -n 3 logs/${m}_int8_build.log | cut -c1-170
  if [ $rc -eq 0 ]; then $TRT --loadEngine=engines/${m}_int8.engine $BENCH > logs/${m}_int8.log 2>&1; fi
  echo "--- $m done $(date +%T) ---"
done
echo
echo "=================== COMPARISON (GPU compute time per image, batch 1, MAXN) ==================="
printf "%-32s %-6s %10s %10s %12s\n" model prec "mean ms" "median ms" "throughput"
for m in $MODELS; do
  for p in fp16 int8; do
    L=logs/${m}_$p.log
    if [ -f "$L" ] && grep -q "GPU Compute Time: min" "$L"; then
      mean=$(grep "GPU Compute Time: min" "$L" | sed -E 's/.*mean = ([0-9.]+) ms.*/\1/')
      med=$(grep "GPU Compute Time: min" "$L" | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
      qps=$(grep "Throughput:" "$L" | sed -E 's/.*Throughput: ([0-9.]+) qps.*/\1/')
      printf "%-32s %-6s %10s %10s %9s qps\n" "$m" "$p" "$mean" "$med" "$qps"
    else
      printf "%-32s %-6s %s\n" "$m" "$p" "(no result - build failed or not run)"
    fi
  done
done
echo "ALL DONE $(date +%T)"