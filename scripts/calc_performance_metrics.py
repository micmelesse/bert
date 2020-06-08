#!/usr/bin/python3

import sys, traceback

if len(sys.argv) <= 1:
    print("Please specify the run_output file")
    exit(-1)

seq_len = 128
if len(sys.argv) > 2:
    seq_len = int(sys.argv[2])
batch_size = 1
if len(sys.argv) > 3:
    batch_size = int(sys.argv[3])
ngpus = 1
if len(sys.argv) > 4:
    ngpus = int(sys.argv[4])

try:
    with open(sys.argv[1]) as f:
        sum = 0
        count = 0
        for l in f.readlines():
            if not "global_step/sec" in l:
                continue
            if not "tensorflow" in l:
                continue
            val = float(l.split(' ')[-1])
            sum += val
            count += 1
        #print(sum, count)
        print("Average global steps per second: {:.3f}".format(sum / count))
        print("Average tokens per second: {:.2f}".format(sum / count * seq_len * batch_size * ngpus))
        print("    given sequence length {}, batch size {}, and {} GPU(s) used".format(seq_len, batch_size, ngpus))
except:
    #traceback.print_exc()
    print("Error opening run_output file", sys.argv[1])
