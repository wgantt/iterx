#!/bin/bash
export CONDAROOT=/home/wgantt/miniconda3
export PATH=$CONDAROOT/condabin:$PATH
source $HOME/.bashrc

SETTINGS=(mono_uncorrected mono_corrected bi)
LANGS=(ar fa ko ru zh)

declare -A LANG_KEYS
LANG_KEYS[ar]=ara
LANG_KEYS[fa]=fas
LANG_KEYS[ko]=kor
LANG_KEYS[ru]=rus
LANG_KEYS[zh]=zho

CHECKPOINTS_DIR=/brtx/604-nvme2/wgantt/better/checkpoints/multimuc/iterx/
ANNOTATIONS_DIR=/brtx/601-nvme1/wgantt/multimuc/data/annotations/
OUTPUT_DIR=/brtx/601-nvme1/wgantt/multimuc/model_outputs/iterx/

ITERX_DIR=/brtx/601-nvme1/wgantt/iterx
GTT_DIR=/brtx/601-nvme1/wgantt/gtt-fork

for lang in "${LANGS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    conda activate sf-dev
    mkdir -p $OUTPUT_DIR/$setting/$lang
	cd $ITERX_DIR
	PYTHONPATH=./src python scripts/muc/convert_iterx_predictions_to_gtt_templates.py \
	  $CHECKPOINTS_DIR/$setting/$lang/${LANG_KEYS[$lang]}_test_run1.agg.filtered.json \
	  $ANNOTATIONS_DIR/$lang/json/untokenized/${LANG_KEYS[$lang]}_test_run1.agg.filtered.json \
	  $OUTPUT_DIR/$setting/$lang/preds_iterx.out
	echo '-----------------------'
	echo $lang $setting 'RME'
	echo '-----------------------'
	PYTHONPATH=./src python scripts/ceaf-scorer.py score --dataset MUC $OUTPUT_DIR/$setting/$lang/preds_iterx.out $ANNOTATIONS_DIR/$lang/json/untokenized/${LANG_KEYS[$lang]}_test_run1.agg.filtered.json --file-type GTT | tee $OUTPUT_DIR/$setting/$lang/eval_iterx_ceaf_rme.out

	
	echo '-----------------------'
	echo $lang $setting 'REE'
	echo '-----------------------'
	cd $GTT_DIR
	conda activate gtt
	if [ $lang == "zh" ]; then
	  python eval.py --pred_file $OUTPUT_DIR/$setting/$lang/preds_iterx.out --gold_file $ANNOTATIONS_DIR/$lang/json/untokenized/${LANG_KEYS[$lang]}_test_run1.agg.filtered.json --chinese | tee $OUTPUT_DIR/$setting/$lang/eval_iterx_ceaf_ree.out
	else
	  python eval.py --pred_file $OUTPUT_DIR/$setting/$lang/preds_iterx.out --gold_file $ANNOTATIONS_DIR/$lang/json/untokenized/${LANG_KEYS[$lang]}_test_run1.agg.filtered.json | tee $OUTPUT_DIR/$setting/$lang/eval_iterx_ceaf_ree.out
	fi
	conda deactivate
  done
done