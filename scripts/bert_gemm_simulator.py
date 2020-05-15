import math
import argparse
import json

# command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--seq_length', type=int, default=512)
parser.add_argument('--max_prediction_per_seq', type=int,
                    default=77)
parser.add_argument('--num_train_steps', type=int, default=10000)
parser.add_argument('--max_eval_steps', type=int, default=100)

# config file paramters
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--intermediate_size', type=int, default=4096)
parser.add_argument('--num_attention_heads', type=int, default=16)
parser.add_argument('--type_vocab_size', type=int, default=2)
parser.add_argument('--vocab_size', type=int, default=30522)
parser.add_argument('--num_hidden_layers', type=int, default=24)


parser.add_argument('--bert_config_file', type=str, default=None)

args = parser.parse_args()

# command line parameters
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
seq_length = args.seq_length
max_prediction_per_seq = args.max_prediction_per_seq
num_train_steps = args.num_train_steps
max_eval_steps = args.max_eval_steps

if args.bert_config_file:
    config_file = json.load(open(args.bert_config_file))
    hidden_size = config_file['hidden_size']
    intermediate_size = config_file['intermediate_size']
    num_attention_heads = config_file['num_attention_heads']
    type_vocab_size = config_file['type_vocab_size']
    vocab_size = config_file['vocab_size']
    num_hidden_layers = config_file['num_hidden_layers']
else:
    # config file paramters
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_attention_heads = args.num_attention_heads
    type_vocab_size = args.type_vocab_size
    vocab_size = args.vocab_size
    num_hidden_layers = args.num_hidden_layers

# # command line parameters
do_train = True
do_eval = True
# batch_size = 6
# eval_batch_size = 8
# seq_length = 512
# max_prediction_per_seq = int(math.ceil(seq_length*0.15))

# # config file paramters
# hidden_size = 1024
# intermediate_size = 4096
# num_attention_heads = 16
# type_vocab_size = 2
# vocab_size = 30522

# intermediate parameters
total_token_count = batch_size * seq_length
eval_total_token_count = eval_batch_size * seq_length
attention_head_size = int(hidden_size / num_attention_heads)
batch_prediction_length = batch_size * max_prediction_per_seq
eval_batch_prediction_length = eval_batch_size * max_prediction_per_seq

# print(batch_size, seq_length, max_prediction_per_seq, hidden_size, intermediate_size,
#       num_attention_heads, type_vocab_size, vocab_size, total_token_count, attention_head_size)

GEMMs = [
    [
        ("gemm", "f32_r", "N", "N", hidden_size, total_token_count, type_vocab_size, 1,
         hidden_size, type_vocab_size, 0,  hidden_size),
        ("gemm", "f32_r", "N", "T", hidden_size, type_vocab_size,  total_token_count, 1,
         hidden_size, type_vocab_size, 0, hidden_size),
        None,
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

def print_train_gemm_configs(g):
    forward_gemm = g[0]
    backward_gemm = g[1]
    backward_gemm_2 = g[2]

    if forward_gemm[0] == "gemm":
        if forward_gemm:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --ldb {} --beta {} --ldc {}".format(*forward_gemm))
        
        if backward_gemm: 
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --ldb {} --beta {} --ldc {}".format(*backward_gemm))
        
        if backward_gemm_2:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --ldb {} --beta {} --ldc {}".format(*backward_gemm_2))
    elif forward_gemm[0] == "gemm_strided_batched":
        if forward_gemm:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --stride_a {} --ldb {} --stride_b {} --beta {} --ldc {} --stride_c {} --batch_count {}".format(*forward_gemm))
        
        if backward_gemm:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --stride_a {} --ldb {} --stride_b {} --beta {} --ldc {} --stride_c {} --batch_count {}".format(*backward_gemm))
        
        if backward_gemm_2:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --stride_a {} --ldb {} --stride_b {} --beta {} --ldc {} --stride_c {} --batch_count {}".format(*backward_gemm_2))

def print_eval_gemm_configs(g):
    eval_gemm = g[3]

    if eval_gemm[0] == "gemm":
        if eval_gemm:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --ldb {} --beta {} --ldc {}".format(*eval_gemm))
        
        
    elif eval_gemm[0] == "gemm_strided_batched":
        if eval_gemm:
            print("./rocblas-bench -f {} -r {} --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha {} --lda {} --stride_a {} --ldb {} --stride_b {} --beta {} --ldc {} --stride_c {} --batch_count {}".format(*eval_gemm))
        
for i, g in enumerate(GEMMs):
    if do_train:
        for step in range(num_train_steps):
            if i >= 1 and i <= 8:
                for layer in range(num_hidden_layers):
                    print_train_gemm_configs(g)
            else:
                print_train_gemm_configs(g)

    if do_eval:
        for step in range(max_eval_steps):
            if i >= 1 and i <= 8:
                for layer in range(num_hidden_layers):
                    print_eval_gemm_configs(g)
            else:
                print_eval_gemm_configs(g)

   