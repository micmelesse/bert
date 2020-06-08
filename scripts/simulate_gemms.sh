pip3 install pandas

SEQ=512
# calculate max prediction per seq
MASKED_LM_PROB=0.15
calc_max_pred() {
  echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
}
MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

python3 scripts/bert_gemm_simulator.py \
    --batch_size=6 \
    --eval_batch_size=8 \
    --seq_length=$SEQ \
    --max_prediction_per_seq=$MAX_PREDICTION_PER_SEQ \
    --hidden_size=1024 \
    --intermediate_size=4096 \
    --num_attention_heads=16 \
    --type_vocab_size=2 \
    --vocab_size=30522 \
    --num_hidden_layers=24 \
    --num_train_steps=10000 \
    --max_eval_steps=100 \
    >bert_gemms_simulated.txt

# python3 scripts/bert_gemm_simulator.py \
#     --batch_size=6 \
#     --eval_batch_size=8 \
#     --seq_length=512 \
#     --max_prediction_per_seq=77 \
#     --bert_config_file="configs/bert_base/bert_config.json" \
#     --num_train_steps=10000 \
#     --max_eval_steps=100 \
#     >bert_gemms_simulated.txt

python3 scripts/get_rocblas_bench_count.py bert_gemms_simulated.txt
