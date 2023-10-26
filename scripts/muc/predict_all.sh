#!/bin/bash

SETTINGS=(mono_uncorrected mono_corrected bi)
LANGS=(ar fa ko ru zh)

CHECKPOINTS_DIR=/brtx/604-nvme2/wgantt/better/checkpoints/multimuc/iterx/


for lang in ${LANGS[@]}; do
	for setting in ${SETTINGS[@]}; do
		sbatch scripts/muc/predict_iterx.sh \
		$CHECKPOINTS_DIR/$setting/$lang \
		$lang \
		$setting
  done
done