DATA_DIR=./data/wikipedia
mkdir -p $DATA_DIR
cd $DATA_DIR

# download wikipedia
wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2


# extract wikipedia
bzip2 -dkv enwiki-latest-pages-articles.xml.bz2
git clone https://github.com/attardi/wikiextractor
python3 wikiextractor/WikiExtractor.py -o wiki_text enwiki-latest-pages-articles.xml

# set seq length
SEQ=128

# calculate max prediction per seq
MASKED_LM_PROB=0.15
calc_max_pred() {
    echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
}
MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

# setup dirs
DATA_DIR=./data/wikipedia
WIKI_TEXT_DIR=${DATA_DIR}/wiki_text
WIKI_TFRECORD_DIR=$DATA_DIR/wiki_tfrecord_seq${SEQ}

mkdir -p $WIKI_TFRECORD_DIR

# generate tfrecord of data in parallel
for DIR in ${WIKI_TEXT_DIR}/*; do
    for FILE in $DIR/*; do
        DIR_BASENAME=$(basename $DIR)
        FILE_BASENAME=$(basename $FILE)
        python3 create_pretraining_data.py \
            --input_file=${FILE} \
            --output_file=$WIKI_TFRECORD_DIR/${DIR_BASENAME}--${FILE_BASENAME}.tfrecord \
            --vocab_file=configs/bert_large/vocab.txt \
            --do_lower_case=True \
            --max_seq_length=${SEQ} \
            --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
            --masked_lm_prob=$MASKED_LM_PROB \
            --random_seed=12345 \
            --dupe_factor=5 &
        sleep 1
    done
done

wait
echo "Done preprocessing wikipedia"
