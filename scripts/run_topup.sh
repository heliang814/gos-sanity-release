#!/usr/bin/env bash
# 串行补 trial: pedestrian × 4 bundle × 3 + protein gos_original×1 + protein replace_similar×2
set -u
cd /mnt/d/Gos/gos-sanity
source .venv/bin/activate
LOG=/mnt/c/Users/maxfo/Desktop/gos-sanity-results/topup.log
echo "=== topup start $(date -Iseconds) ===" | tee -a "$LOG"

run() {
  local tag="$1"; shift
  echo "--- [$tag] $(date -Iseconds) :: $*" | tee -a "$LOG"
  python -m scripts.run_experiment "$@" 2>&1 | tee -a "$LOG"
  echo "--- [$tag] done rc=$? $(date -Iseconds)" | tee -a "$LOG"
}

for r in 1 2 3; do
  run "ped_round$r" --queries data/queries_pedestrian.json
done
run "pro_gos_rep" --queries data/queries_protein.json --bundle-types gos_original,replace_similar
run "pro_rep_only" --queries data/queries_protein.json --bundle-types replace_similar

echo "=== topup all done $(date -Iseconds) ===" | tee -a "$LOG"
