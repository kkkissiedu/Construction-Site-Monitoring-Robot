cd ~/crackseg
for f in yolo26n-seg_crack_raw yolo26s-seg_crack_raw; do
  for c in 0.25 0.1; do
    echo; echo "##### $f conf=$c"
    CONF=$c python3 eval_seg.py heldout/images heldout/labels engines/${f}_fp16.engine engines/${f}_int8pin.engine 2>&1 | tail -7
  done
done