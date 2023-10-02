#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1
#SBATCH --time=24:0:0

# 1: Jsonnet config file for training
# 2: Model serialization directory
export CONDAROOT=/home/wgantt/miniconda3
export PATH=$CONDAROOT/condabin:$PATH
source $HOME/.bashrc
export MKL_THREADING_LAYER=GNU

conda activate sf-dev
PYTHONPATH=./src allennlp train \
  $1 \
  --include-package iterx \
  -s $2