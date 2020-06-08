#!/bin/bash
set -e

# use bash
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR_PATH=$(dirname $SCRIPT_PATH)
if [ ! "$BASH_VERSION" ]; then
  echo "Using bash to run this script $0" 1>&2
  exec bash $SCRIPT_PATH "$@"
  exit 1
fi

# process commandline args

# choose arch
if [[ "$*" == *"base"* ]]; then
  echo "Running BERT Base"
  MODEL_CONFIG_DIR=configs/bert_base
  TRAIN_DIR=bert_base
else
  echo "Running BERT Large"
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

# choose hardware
if [[ "$*" == *"nvidia"* ]]; then
  echo "Running on Nvidia"
  TRAIN_DIR="${TRAIN_DIR}_nvidia"
else
  echo "Running on AMD"
  TRAIN_DIR="${TRAIN_DIR}_amd"
fi

# choose precision
if [[ "$*" == *"fp16"* ]]; then
  echo "Running FP16"
  TRAIN_DIR="${TRAIN_DIR}_fp16"
  PREC=fp16
else
  echo "Running FP32"
  TRAIN_DIR="${TRAIN_DIR}_fp32"
  PREC=fp32
fi

# collect rpt summary
if [[ "$*" == *"rpt"* ]]; then
  echo "Collecting RPT summary"
  export HCC_PROFILE=2
  TRAIN_DIR="${TRAIN_DIR}_rpt"
fi

# rocblas trace
if [[ "$*" == *"rocblas"* ]]; then
  echo "ROCBLAS Trace"
  export ROCBLAS_LAYER=3
  TRAIN_DIR="${TRAIN_DIR}_rocblas"
fi

# rocblas trace
if [[ "$*" == *"map"* ]]; then
  echo "Mapping GEMMs"
  export ROCBLAS_LAYER=2
  export HCC_SERIALIZE_KERNEL=3
  export HCC_SERIALIZE_COPY=3
  export HCC_PROFILE=2
  TRAIN_STEPS=2
  TRAIN_WARM_STEPS=1
  TRAIN_DIR="${TRAIN_DIR}_gemm_map"
fi

#set num of train steps
if [[ "$*" == *"debug"* ]]; then
  echo "Debug run"

  if [ -n "$TRAIN_STEPS" ]; then
    echo "TRAIN_STEPS set for Debugging"
  fi
  TRAIN_STEPS=10

  if [ -n "$TRAIN_WARM_STEPS" ]; then
    echo "TRAIN_WARM_STEPS set for Debugging"
  fi
  TRAIN_WARM_STEPS=1

  TRAIN_DIR="${TRAIN_DIR}_debug"
else

  if [ -n "$TRAIN_STEPS" ]; then
    echo "TRAIN_STEPS already set"
  else
    TRAIN_STEPS=1000
  fi

  if [ -n "$TRAIN_WARM_STEPS" ]; then
    echo "TRAIN_WARM_STEPS already set"
  else
    TRAIN_WARM_STEPS=100
  fi
fi

# set data dir
DATA_DIR=./data

# create train dir
mkdir -p $TRAIN_DIR

# prep train dir
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

DATA_SOURCE_FILE_PATH=data/wiki_00
DATA_SOURCE_NAME=$(basename "$DATA_SOURCE_FILE_PATH")

# iterate through configs (Batch, Sequence Length)
BATCH=6
SEQ=512

# calculate max prediction per seq
MASKED_LM_PROB=0.15
calc_max_pred() {
  echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
}
MAX_PREDICTION_PER_SEQ=$(calc_max_pred)
echo "MAX_PREDICTION_PER_SEQ: $MAX_PREDICTION_PER_SEQ"

# create config train dir
CUR_TRAIN_DIR=$TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}_${PREC}

# clean prev train dir
rm -rf $CUR_TRAIN_DIR

# create fresh train dir
mkdir -p $CUR_TRAIN_DIR

DATA_TFRECORD=$DATA_DIR/${DATA_SOURCE_NAME}_seq${SEQ}.tfrecord
if [ ! -f "$DATA_TFRECORD" ]; then
  # generate tfrecord of data
  python3 create_pretraining_data.py \
    --input_file=$DATA_SOURCE_FILE_PATH \
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
  export ROCBLAS_LOG_TRACE_PATH=$CUR_TRAIN_DIR/rocblas_log_trace_ba${BATCH}_seq${SEQ}_${PREC}.txt
  export ROCBLAS_LOG_BENCH_PATH=$CUR_TRAIN_DIR/rocblas_log_bench_ba${BATCH}_seq${SEQ}_${PREC}.txt
fi

# rocblas trace
if [[ "$*" == *"map"* ]]; then
  export ROCBLAS_LOG_BENCH_PATH=$CUR_TRAIN_DIR/rocblas_log_bench_ba${BATCH}_seq${SEQ}_${PREC}.txt
fi

# run pretraining
# choose precision
if [[ "$*" == *"fp16"* ]]; then
  python3 run_pretraining_fp16.py \
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
    --auto_mixed_precision=True \
    2>&1 | tee $CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}_${PREC}.txt
else
  python3 run_pretraining.py \
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
  2>&1 | tee $CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}_${PREC}.txt
fi

# rocblas trace
if [[ "$*" == *"rocblas"* ]]; then
  pip3 install pandas
  python3 scripts/get_rocblas_bench_count.py $CUR_TRAIN_DIR/rocblas_log_bench_ba${BATCH}_seq${SEQ}_${PREC}.txt
fi

# model metrics
if [[ "$*" == *"metrics"* ]]; then
  pip3 install pandas
  python3 scripts/get_bert_model_metrics.py $CUR_TRAIN_DIR
fi

# get rpt summary
if [[ "$*" == *"rpt"* ]]; then
  /opt/rocm/hcc/bin/rpt --topn -1 $CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}_${PREC}.txt \
    >$CUR_TRAIN_DIR/${DATA_SOURCE_NAME}_ba${BATCH}_seq${SEQ}_${PREC}_rpt_summary.txt
fi
