#!/bin/bash
# Official LineVul on our audited splits. Outputs go under the PROJECT dir --
# an earlier run wrote to a scratchpad that was wiped between sessions, losing
# ~8h of training and the only copy of its metrics.
R=/home/moses/TSE_Journal
LV=$R/baselines/LineVul/linevul
CSV=$R/data/baseline_csv
OUT=$R/runs/linevul_raw
PY=/home/moses/venv/bin/python
pick(){ while true; do for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 25000 ] && { echo $g; return; }; done; sleep 90; done; }
cd $LV
for m in repo random group; do
  g=$(pick); echo "linevul/$m -> gpu $g"
  LINEVUL_DUMP=$OUT/$m CUDA_VISIBLE_DEVICES=$g $PY -u linevul_main.py \
    --output_dir=$OUT/model_$m --model_name=linevul_$m.bin --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train --do_test \
    --train_data_file=$CSV/$m/train.csv --eval_data_file=$CSV/$m/val.csv \
    --test_data_file=$CSV/$m/test.csv \
    --epochs 5 --block_size 512 --train_batch_size 32 --eval_batch_size 128 \
    --learning_rate 2e-5 --max_grad_norm 1.0 --seed 42 > $OUT/$m.log 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "FAILED linevul/$m rc=$rc" || echo "OK linevul/$m"
done
echo LINEVUL_DONE
