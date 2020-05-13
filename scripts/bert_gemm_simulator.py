import math

# command line parameters
do_train = True
do_eval = True
batch_size = 6
eval_batch_size = 8
seq_length = 512
max_prediction_per_seq = int(math.ceil(seq_length*0.15))

# config file paramters
hidden_size = 1024
intermediate_size = 4096
num_attention_heads = 16
type_vocab_size = 2
vocab_size = 30522

# intermediate parameters
total_token_count = batch_size * seq_length
eval_total_token_count = eval_batch_size * seq_length
attention_head_size = int(hidden_size / num_attention_heads)
batch_prediction_length = batch_size * max_prediction_per_seq
eval_batch_prediction_length = eval_batch_size * max_prediction_per_seq

print(batch_size, seq_length, max_prediction_per_seq, hidden_size, intermediate_size,
      num_attention_heads, type_vocab_size, vocab_size, total_token_count, attention_head_size)

GEMMs = [
    [
        ("gemm", "f32_r", "N", "N", hidden_size, total_token_count, type_vocab_size, 1,
         hidden_size, type_vocab_size, 0,  hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, type_vocab_size,  total_token_count, 1,
         hidden_size, type_vocab_size, 0, hidden_size),
        ("gemm", "f32_r", "T", "N", type_vocab_size, batch_size, hidden_size, 1,
         hidden_size, hidden_size, 0, type_vocab_size),
        ("gemm", "f32_r", "N", "N",  hidden_size, eval_total_token_count,
         type_vocab_size, 1, hidden_size, type_vocab_size, 0, hidden_size)
    ],
    [
        ("gemm", "f32_r", "N", "N", hidden_size, total_token_count, hidden_size, 1,
         hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, hidden_size, total_token_count, 1,
         hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "T",  "N", hidden_size, total_token_count,
         hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "N", hidden_size,  eval_total_token_count,
         hidden_size,  1, hidden_size, hidden_size,  0,  hidden_size)
    ], [
        ("gemm", "f32_r", "N", "N", hidden_size, total_token_count, hidden_size, 1,
         hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, hidden_size, total_token_count, 1,
         hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "T",  "N", hidden_size, total_token_count,
         hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "N", hidden_size,  eval_total_token_count,
         hidden_size,  1, hidden_size, hidden_size,  0,  hidden_size)
    ], [
        ("gemm", "f32_r", "N", "N", hidden_size, total_token_count, hidden_size, 1,
         hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, hidden_size, total_token_count, 1,
         hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "T",  "N", hidden_size, total_token_count,
         hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "N", hidden_size,  eval_total_token_count,
         hidden_size,  1, hidden_size, hidden_size,  0,  hidden_size)
    ],
    [
        ("gemm_strided_batched", "f32_r", "T", "N", seq_length, seq_length, attention_head_size, 1, attention_head_size, seq_length*attention_head_size,
         attention_head_size, seq_length*attention_head_size, 0, seq_length, seq_length*seq_length, batch_size*num_attention_heads),
        ("gemm_strided_batched", "f32_r", "N", "N", attention_head_size, seq_length, seq_length, 1, attention_head_size, seq_length*attention_head_size,
            seq_length, seq_length*seq_length, 0, attention_head_size, seq_length*attention_head_size, batch_size*num_attention_heads),
        ("gemm_strided_batched", "f32_r", "N", "T", attention_head_size, seq_length, seq_length, 1, attention_head_size,  seq_length*attention_head_size,
            seq_length, seq_length*seq_length, 0, attention_head_size, seq_length*attention_head_size, batch_size*num_attention_heads),
        ("gemm_strided_batched",  "f32_r", "T", "N",  seq_length,  seq_length,  attention_head_size, 1,  attention_head_size, seq_length *
            attention_head_size, attention_head_size, seq_length*attention_head_size, 0, seq_length, seq_length*seq_length, eval_batch_size*num_attention_heads)
    ],
    [("gemm_strided_batched", "f32_r", "N", "N", attention_head_size, seq_length, seq_length, 1, attention_head_size, seq_length*attention_head_size,
      seq_length, seq_length*seq_length, 0, attention_head_size, seq_length*attention_head_size, batch_size*num_attention_heads),
        ("gemm_strided_batched", "f32_r", "N", "T", attention_head_size, seq_length, seq_length, 1, attention_head_size, seq_length*attention_head_size,
         seq_length, seq_length*seq_length, 0, attention_head_size, seq_length*attention_head_size, batch_size*num_attention_heads),
        ("gemm_strided_batched", "f32_r", "T", "N", seq_length, seq_length, attention_head_size, 1, attention_head_size, seq_length*attention_head_size,
         attention_head_size, seq_length*attention_head_size, 0, seq_length, seq_length*seq_length, batch_size*num_attention_heads),
        ("gemm_strided_batched", "f32_r", "N", "N", attention_head_size, seq_length, seq_length, 1, attention_head_size, seq_length*attention_head_size,
         seq_length, seq_length*seq_length, 0, attention_head_size, seq_length*attention_head_size, eval_batch_size*num_attention_heads)
     ],
    [("gemm", "f32_r", "N", "N", hidden_size, total_token_count, hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
     ("gemm", "f32_r", "N", "T", hidden_size, hidden_size,
      total_token_count, 1, hidden_size, hidden_size, 0, hidden_size),
     ("gemm", "f32_r", "T", "N",  hidden_size, total_token_count,
      hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
     ("gemm", "f32_r", "N", "N", hidden_size,  eval_total_token_count,
        hidden_size,  1, hidden_size, hidden_size,  0,  hidden_size)
     ],
    [("gemm", "f32_r", "N",  "N", intermediate_size, total_token_count, hidden_size, 1,
      intermediate_size, hidden_size, 0, intermediate_size),
     ("gemm", "f32_r",  "N", "T", hidden_size, intermediate_size,
      total_token_count, 1, hidden_size, intermediate_size, 0, hidden_size),
     ("gemm", "f32_r", "T", "N", intermediate_size, total_token_count, hidden_size, 1, hidden_size, hidden_size, 0, intermediate_size
      ),
     ("gemm", "f32_r",  "N",  "N",  intermediate_size,  eval_total_token_count,
      hidden_size,  1,  intermediate_size,  hidden_size,  0, intermediate_size)
     ],
    [("gemm", "f32_r", "N", "N", hidden_size, total_token_count, intermediate_size, 1, hidden_size, intermediate_size, 0, hidden_size),
     ("gemm", "f32_r", "N", "T", intermediate_size, hidden_size,
      total_token_count, 1, intermediate_size, hidden_size, 0, intermediate_size),
     ("gemm", "f32_r", "T", "N",  hidden_size, total_token_count,
      intermediate_size, 1, intermediate_size, intermediate_size, 0, hidden_size),
     ("gemm", "f32_r",  "N",  "N",  hidden_size, eval_total_token_count, intermediate_size, 1, hidden_size, intermediate_size, 0, hidden_size)],
    [
        ("gemm", "f32_r", "N", "N", hidden_size, batch_size,
         hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, hidden_size,
         batch_size, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "T", "N", hidden_size, batch_size,
         hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "N", hidden_size, eval_batch_size,
         hidden_size, 1, hidden_size, hidden_size, 0, hidden_size)
    ],
    [
        ("gemm", "f32_r", "N", "N", hidden_size, batch_prediction_length,
         hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, hidden_size,
         batch_prediction_length, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "T", "N", hidden_size, batch_prediction_length,
         hidden_size, 1, hidden_size, hidden_size, 0, hidden_size),
        ("gemm", "f32_r", "N",  "N", hidden_size,  eval_batch_prediction_length,
         hidden_size,  1,  hidden_size,  hidden_size,  0,  hidden_size)
    ],
    [
        ("gemm", "f32_r", "T", "N", vocab_size, batch_prediction_length,
         hidden_size, 1, hidden_size, hidden_size, 0, vocab_size),
        ("gemm", "f32_r", "N", "N", hidden_size, batch_prediction_length,
         vocab_size, 1, hidden_size, vocab_size, 0, hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, vocab_size,
         batch_prediction_length, 1, hidden_size, vocab_size, 0, hidden_size),
        ("gemm", "f32_r", "T", "N", vocab_size, eval_batch_prediction_length, hidden_size, 1, hidden_size, hidden_size, 0, vocab_size)],
    [
        ("gemm", "f32_r", "T", "N", 2, batch_size,
         hidden_size, 1, hidden_size, hidden_size, 0, 2),
        ("gemm", "f32_r", "N", "N", hidden_size,
         batch_size, 2, 1, hidden_size, 2, 0, hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, 2,
         batch_size, 1, hidden_size, 2, 0, hidden_size),
        ("gemm", "f32_r",  "T", "N",  2, eval_batch_size, hidden_size, 1, hidden_size, hidden_size, 0, 2)]

]

for g in GEMMs:
    forward_gemm = g[0]
    backward_gemm = g[1]
    backward_gemm_2 = g[2]
    eval_gemm = g[3]

    if forward_gemm[0] == "gemm":
        if do_train:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --ldb {} --beta {} --ldc {}".format(*forward_gemm))
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --ldb {} --beta {} --ldc {}".format(*backward_gemm))
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --ldb {} --beta {} --ldc {}".format(*backward_gemm_2))

        if do_eval:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --ldb {} --beta {} --ldc {}".format(*eval_gemm))

    elif forward_gemm[0] == "gemm_strided_batched":

        if do_train:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --stride_a {} --ldb {} --stride_b {} --beta {} --ldc {} --stride_c {} --batch_count {}".format(*forward_gemm))
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --stride_a {} --ldb {} --stride_b {} --beta {} --ldc {} --stride_c {} --batch_count {}".format(*backward_gemm))
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --stride_a {} --ldb {} --stride_b {} --beta {} --ldc {} --stride_c {} --batch_count {}".format(*backward_gemm_2))

        if do_eval:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --stride_a {} --ldb {} --stride_b {} --beta {} --ldc {} --stride_c {} --batch_count {}".format(*eval_gemm))
