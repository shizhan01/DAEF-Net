#!/usr/bin/env bash
# Data generation, training over five seeds and the robustness tests for D2 and D4.
set -e

python scripts/generate_d2.py --out data/d2
python scripts/generate_d2.py --out data/d2_offset1ms --splits test --clock-offset-ms 1
python scripts/generate_d2.py --out data/d2_offset2ms --splits test --clock-offset-ms 2
python scripts/generate_d2.py --out data/d2_offset5ms --splits test --clock-offset-ms 5
python scripts/generate_d2.py --out data/d2_49p5hz --splits test --freq 49.5
python scripts/generate_d2.py --out data/d2_50p5hz --splits test --freq 50.5
python scripts/generate_d4.py --out data/d4

for seed in 0 1 2 3 4; do
  python scripts/train.py --config configs/d2.yaml --data data/d2 --out runs/d2_s${seed} --seed ${seed}
  python scripts/evaluate.py --ckpt runs/d2_s${seed}/best.pt --data data/d2/test.npz --test-snr 20 \
      --out runs/d2_s${seed}/test_20db.json
  for d in d2_offset1ms d2_offset2ms d2_offset5ms d2_49p5hz d2_50p5hz; do
    python scripts/evaluate.py --ckpt runs/d2_s${seed}/best.pt --data data/${d}/test.npz \
        --out runs/d2_s${seed}/${d}.json
  done

  python scripts/train.py --config configs/d4.yaml --data data/d4 --out runs/d4_s${seed} --seed ${seed}
  python scripts/robustness.py --ckpt runs/d4_s${seed}/best.pt --data data/d4/test.npz \
      --out runs/d4_s${seed}/robustness.json
done
