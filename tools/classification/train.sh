ROOT=..
GPU=$1
LANG=$2
SEED=$3
TASK=MARC-2
DATA=$ROOT/data/$TASK/$LANG
MODEL=bert-base-uncased
FILE_FORMAT=csv
SAVE_DIR=$ROOT/outputs/$LANG/${TASK}/${MODEL}/seed${SEED}

mkdir -p $SAVE_DIR
# SAVE_FILE=$SAVE_DIR/model.pt
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --model $MODEL \
    --train $DATA/train.csv \
    --valid $DATA/validation.csv \
    --test $DATA/test.csv \
    --format $FILE_FORMAT \
    --batch_size 32 \
    --max_epoch 5 \
    --warmup 0.1 \
    --learning_rate 5e-05 \
    --early_stopping 10 \
    --seed $SEED \
    --save_dir $SAVE_DIR > $SAVE_DIR/train.log
