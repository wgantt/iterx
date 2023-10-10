#!/bin/bash
# Produces CEAF-RME scores for GTT-formatted predictions

SETTINGS=(bi mono_corrected mono_uncorrected)
LANGS=(ar fa ko ru zh)

declare -A LANG_KEYS
LANG_KEYS[ar]=ara
LANG_KEYS[fa]=fas
LANG_KEYS[ko]=kor
LANG_KEYS[ru]=rus
LANG_KEYS[zh]=zho

PREDICTIONS_ROOT=/brtx/601-nvme1/wgantt/multimuc/model_outputs/gtt
GOLD_ROOT=/brtx/601-nvme1/wgantt/multimuc/data/annotations

conda activate sf-dev

for lang in "${LANGS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    PRED_FILE=$PREDICTIONS_ROOT/$setting/$lang/preds_gtt.out
	GOLD_FILE=$GOLD_ROOT/$lang/json/untokenized/${LANG_KEYS[$lang]}_test_run1.agg.filtered.json
	echo '-----------------------'
	echo $lang $setting
	echo '-----------------------'
	PYTHONPATH=./src python scripts/ceaf-scorer.py score --dataset MUC $PRED_FILE $GOLD_FILE --file-type GTT | tee $PREDICTIONS_ROOT/$setting/$lang/eval_gtt_ceaf_rme.out
  done
done