#!/bin/bash

SETTINGS=(bi mono_corrected mono_uncorrected)
LANGS=(ar fa ko ru zh)

CHECKPOINTS_DIR=/brtx/604-nvme2/wgantt/better/checkpoints/multimuc/iterx/


# for lang in "${LANGS[@]}"; do
#   for setting in bi; do
for lang in fa; do
	for setting in bi; do
	sbatch scripts/muc/predict_iterx.sh \
	  $CHECKPOINTS_DIR/$setting/$lang \
	  $lang \
	  $setting
  done
done