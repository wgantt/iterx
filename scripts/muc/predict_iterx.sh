#!/bin/bash
#SBATCH --partition=brtx6-ir
#SBATCH --gpus=1
#SBATCH --time=24:0:0

# 1: Model archive
# 2: Language
# 3: Setting (bi, mono_corrected, mono_uncorrected)
export CONDAROOT=/home/wgantt/miniconda3
export PATH=$CONDAROOT/condabin:$PATH
source $HOME/.bashrc
export MKL_THREADING_LAYER=GNU

MODEL_ARCHIVE=$1
LANG=$2
SETTING=$3


declare -A LANG_KEYS
LANG_KEYS[ar]=ara
LANG_KEYS[fa]=fas
LANG_KEYS[ko]=kor
LANG_KEYS[ru]=rus
LANG_KEYS[zh]=zho

export INPUT_FILE=${LANG_KEYS[$LANG]}_test_run1.agg.filtered.json
export INPUT_PATH=/brtx/601-nvme1/wgantt/multimuc/data/annotations/$LANG/json/sf-outputs/mono_corrected/$LANG/$INPUT_FILE
export REF_PATH=/brtx/601-nvme1/wgantt/multimuc/data/annotations/$LANG/json/untokenized/$INPUT_FILE
export OUTPUT_PATH=$MODEL_ARCHIVE/${LANG_KEYS[$LANG]}_test_run1.agg.filtered.v3.json

conda activate sf-dev
PYTHONPATH=./src allennlp predict \
  $MODEL_ARCHIVE/model.tar.gz \
  $INPUT_PATH \
  --output-file $OUTPUT_PATH \
  --include-package iterx \
  --use-dataset-reader \
  --cuda-device 0 \
  --overrides '{"model.metrics.muc.doc_path": {"'"$INPUT_PATH"'": "'"$REF_PATH"'"}}'