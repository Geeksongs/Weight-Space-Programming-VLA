#!/bin/bash
# usage: par_run.sh SUITE TASK EPS "edit1 edit2 edit3"   -> runs up to 3 candidates in parallel, prints success + grasp proxy
cd /media/python/Seagate/research/smovla_astra
PY=/home/python/miniconda3/envs/lerobot/bin/python
SUITE=$1; TASK=$2; EPS=$3; shift 3
for c in $@; do
  $PY code/forward_tool.py --suite $SUITE --task $TASK --eps $EPS --edit runs/run1/edit_$c --tag ${SUITE}_t${TASK}_$c > runs/run1/log_$c.txt 2>&1 &
done
wait
for c in $@; do echo "$c: $(grep -h SUCCESS runs/run1/log_$c.txt | cut -c1-14) $(grep -h -E 'rror' runs/run1/log_$c.txt | tail -1 | cut -c1-80)"; done
