#!/bin/bash

# use bash
SCRIPT_PATH=$(realpath $0)
if [ ! "$BASH_VERSION" ]; then
  echo "Using bash to run this script $0" 1>&2
  exec bash $SCRIPT_PATH "$@"
  exit 1
fi

# process commandline args
if [[ "$*" == *"rpt"* ]]; then
  echo "Collecting RPT summary"
  export HCC_PROFILE=2
fi
# choose arch
if [[ "$*" == *"base"* ]]; then
  MODEL_CONFIG_DIR=configs/bert_base
  TRAIN_DIR=bert_base
else
  MODEL_CONFIG_DIR=configs/bert_large
  TRAIN_DIR=bert_large
 
fi

# check if multi gpu
if [[ "$*" == *"mgpu"* ]]; then
  echo "Multi GPU run"
  TRAIN_DIR="${TRAIN_DIR}_mgpu"
else
  echo "Single GPU run"
  # choose a single gpu
  if [[ "$*" == *"nvidia"* ]]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    export HIP_VISIBLE_DEVICES=0
  fi

  TRAIN_DIR="${TRAIN_DIR}_sgpu"
fi

if [[ "$*" == *"nvidia"* ]]; then
    TRAIN_DIR="${TRAIN_DIR}_nvidia"
else
    TRAIN_DIR="${TRAIN_DIR}_amd"
fi

#set num of train steps
if [[ "$*" == *"debug"* ]]; then
  echo "Debug run"
  TRAIN_STEPS=10
  TRAIN_WARM_STEPS=1
  TRAIN_DIR="${TRAIN_DIR}_debug"
else
  TRAIN_STEPS=1500
  TRAIN_WARM_STEPS=150
fi

CODE_DIR=.
DATA_DIR=./data

rm -rf $TRAIN_DIR
mkdir -p $TRAIN_DIR

# prep train dir
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

DATA_SOURCE_FILE_PATH=data/wiki_00
DATA_SOURCE_NAME=$(basename "$DATA_SOURCE_FILE_PATH")

# iterate through configs (Batch, Sequence Length)
BATCH=4
SEQ=512
# calculate max prediction per seq
  MASKED_LM_PROB=0.15
  calc_max_pred() {
    echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
  }
  MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

# create config train dir
CUR_TRAIN_DIR=$TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}
mkdir -p $CUR_TRAIN_DIR

DATA_TFRECORD=$DATA_DIR/${DATA_SOURCE_NAME}_seq${SEQ}.tfrecord
if [ ! -f "$DATA_TFRECORD" ]; then
  # generate tfrecord of data
  python3 $CODE_DIR/create_pretraining_data.py \
    --input_file=$CODE_DIR/$DATA_SOURCE_FILE_PATH \
    --output_file=$DATA_TFRECORD \
    --vocab_file=$TRAIN_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=$SEQ \
    --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
    --masked_lm_prob=$MASKED_LM_PROB \
    --random_seed=12345 \
    --dupe_factor=5
fi

# rocblas trace
if [[ "$*" == *"rocblas"* ]]; then
  echo "ROCBLAS Trace"
  export ROCBLAS_LAYER=3
  export ROCBLAS_LOG_TRACE_PATH=$CUR_TRAIN_DIR/rocblas_log_trace.txt
  export ROCBLAS_LOG_BENCH_PATH=$CUR_TRAIN_DIR/rocblas_log_bench.txt
fi

# run pretraining
python3 $CODE_DIR/run_pretraining.py \
  --input_file=$DATA_TFRECORD \
  --output_dir=$CUR_TRAIN_DIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$TRAIN_DIR/bert_config.json \
  --train_batch_size=$BATCH \
  --max_seq_length=$SEQ \
  --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
  --num_train_steps=$TRAIN_STEPS \
  --num_warmup_steps=$TRAIN_WARM_STEPS \
  --learning_rate=1e-4 \
  2>&1 | tee $CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}.txt

# get rpt summary
if [[ "$*" == *"rpt"* ]]; then
  /opt/rocm/hcc/bin/rpt --topn -1  $CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}.txt \
  >$CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}_rpt_summary.txt
fi