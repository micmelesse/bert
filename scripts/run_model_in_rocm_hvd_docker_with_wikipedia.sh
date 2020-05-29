#!/bin/bash

# The following may be modified
# Model can be "bert_large" or "bert_base"
MODEL="bert_large"
# Sequence length can be 128, 256, 512, or other number of choice
SEQ=128
# Batch size can be anything that fits the GPU
BATCH=40
# NP is the number of GPUs used. Set to equal or less than the number of GPUs on system.
NP=8

# Container image used
IMAGE="rocm/tensorflow:rocm3.3-tf2.1-ofed4.6-openmpi4.0.0-horovod"

# Print a message
echo "This script will run the $MODEL model in a ROCm container. "
echo "This is done with multi-GPU using Horovod."
echo "Below is what it will do:"
echo "  1. Pull the latest ROCm docker image;"
echo "  2. Check that wikipedia data set exists"
echo "  3. Train $MODEL inside the ROCm container;"
echo "  4. Clean up the container."
echo "Please press any key to start, or ESC to exit."

# Read a key press
read -n 1 -s KEY
if [ "$KEY" == $'\e' ] ; then
  echo "Exit. Did nothing."
  exit 0
fi

# Get the folders
SCRIPTPATH=$(dirname $(realpath $0))
CODE_DIR=$SCRIPTPATH/..
CODE_DIR_INSIDE=/data/code

# Pull the docker image
echo 
echo "=== Docker pulling image $IMAGE and start container ==="
docker pull $IMAGE
CTNRNAME=ROCmDockerContainer
echo -n "Is $CTNRNAME running? "
docker inspect -f '{{.State.Running}}' $CTNRNAME
if [ $? -eq 0 ]; then
    echo -n "Container $CTNRNAME is running. Stopping first ... "
    docker stop $CTNRNAME
else
    echo "An \"Error\" message here is normal. It just indicates that the container is not currently running (as expected)."
fi
echo "Starting $CTNRNAME"
docker run --name $CTNRNAME -it -d --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --user $(id -u):$(id -g) -w $CODE_DIR_INSIDE -v $CODE_DIR:$CODE_DIR_INSIDE $IMAGE

# Preparing temporary folder for training
TRAIN_DIR_NAME=dashboard_train_dir
TRAIN_DIR=$CODE_DIR/$TRAIN_DIR_NAME
rm -rf $TRAIN_DIR
mkdir -p $TRAIN_DIR
TRAIN_DIR_INSIDE=$CODE_DIR_INSIDE/$TRAIN_DIR_NAME

# Copy the configuration file and vocab file
MODEL_CONFIG_DIR=$CODE_DIR/configs/$MODEL
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

# Check training data
echo 
echo "=== Check training data ==="

# Check the training data
WIKI_TFRECORD_DIR=data/wikipedia/wiki_tfrecord_seq${SEQ}
if [ ! -d "$WIKI_TFRECORD_DIR" ]; then
  echo "No preprocessed wikipedia data found for seq length ${SEQ}"
  echo "Use the script get_wikipedia_dataset.sh"
  exit
fi

# calculate max prediction per seq
MASKED_LM_PROB=0.15
calc_max_pred() {
  echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
}
MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

# Perform training
echo 
echo "=== Training BERT ==="
CUR_TRAINING=wikipedia_ba${BATCH}_seq${SEQ}
mkdir -p $TRAIN_DIR/$CUR_TRAINING
OUTPUT_FILE_REL=$CUR_TRAINING/$CUR_TRAINING.txt

TRAIN_STEPS=1000000
WARMUP_STEPS=10000
LEARNING_RATE=2e-5

# export HIP_VISIBLE_DEVICES=0 # choose gpu
# run pretraining
docker exec $CTNRNAME \
mpirun -np $NP \
  -H localhost:$NP \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO \
  -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x LD_LIBRARY_PATH -x PATH \
  -mca pml ob1 -mca btl ^openib \
python3 $CODE_DIR_INSIDE/run_pretraining.py \
  --input_file=$WIKI_TFRECORD_DIR/*.tfrecord \
  --output_dir=$TRAIN_DIR_INSIDE/$CUR_TRAINING \
  --do_train=True \
  --do_eval=True \
  --use_horovod=True \
  --bert_config_file=$TRAIN_DIR_INSIDE/bert_config.json \
  --train_batch_size=$BATCH \
  --max_seq_length=$SEQ \
  --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
  --num_train_steps=$TRAIN_STEPS \
  --num_warmup_steps=$WARMUP_STEPS \
  --learning_rate=$LEARNING_RATE \
  2>&1 | tee $TRAIN_DIR/$OUTPUT_FILE_REL

# Calculate performance metrics
echo 
echo "=== Training Performance ==="
if [ ! -f "$OUTPUT_FILE" ]; then
  docker exec $CTNRNAME \
  python3 $CODE_DIR_INSIDE/scripts/calc_performance_metrics.py \
    $TRAIN_DIR_INSIDE/$OUTPUT_FILE_REL $SEQ $BATCH $NP
fi

# Cleaning up
echo 
echo "=== Cleaning up ==="
echo -n "Stopping "
docker stop $CTNRNAME
