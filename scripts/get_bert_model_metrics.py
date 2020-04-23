from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import pandas as pd
import sys
import argparse


def get_event_acc(log_dir):
    event_acc = EventAccumulator(os.path.expanduser(log_dir))
    event_acc.Reload()
    return event_acc


def get_scalar_data(event_acc, scalar_name):
    w_times, steps, vals = zip(*event_acc.Scalars(scalar_name))
    df = pd.DataFrame([list(t) for t in list(zip(steps, vals, w_times))])
    df.columns = ["step", "value", "wall time"]
    return df


parser = argparse.ArgumentParser()
parser.add_argument('bert_train_dir', type=str, help='bert training directory')
args = parser.parse_args()


def check_dir(path):
    for f in os.listdir(path):
        if "tfevents" in f:
            return True
    return False


if args.bert_train_dir:
    if not check_dir(args.bert_train_dir):
        print("Make sure the input directory contains .tfevents file")

    event_acc = get_event_acc(args.bert_train_dir)
    tags = event_acc.Tags()
    scalars = tags["scalars"]

    for s in scalars:

        csv_path = os.path.join(args.bert_train_dir,
                                s.replace("/", "--")+".csv")
        # print("Saving", csv_path)
        s_data = get_scalar_data(event_acc, s)
        print(s+" avg: ",s_data["value"].mean())
        s_data.to_csv(csv_path)
