# choose bert large or bert base
MODEL_CONFIG_DIR=configs/bert_large

# choose bert config
BATCH=6
SEQ=512

# choose data source and preprocessed output
DATA_SOURCE_FILE_PATH=sample_text.txt
DATA_TFRECORD=sample_text_seq${SEQ}.tfrecord

# calculate max prediction per seq
MASKED_LM_PROB=0.15
calc_max_pred() {
  echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
}
MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

# generate tfrecord of data
python3 create_pretraining_data.py \
--input_file=$DATA_SOURCE_FILE_PATH \
--output_file=$DATA_TFRECORD \
--vocab_file=$MODEL_CONFIG_DIR/vocab.txt \
--do_lower_case=True \
--max_seq_length=$SEQ \
--max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
--masked_lm_prob=$MASKED_LM_PROB \
--random_seed=12345 \
--dupe_factor=5


# run pretraining
TRAIN_DIR=bert_large_ba${BATCH}_seq${SEQ}
TRAIN_STEPS=1000
TRAIN_WARM_STEPS=100
LEARNING_RATE=1e-4 

python3 run_pretraining.py \
  --input_file=$DATA_TFRECORD \
  --output_dir=$TRAIN_DIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$MODEL_CONFIG_DIR/bert_config.json \
  --train_batch_size=$BATCH \
  --max_seq_length=$SEQ \
  --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
  --num_train_steps=$TRAIN_STEPS \
  --num_warmup_steps=$TRAIN_WARM_STEPS \
  --learning_rate=$LEARNING_RATE 