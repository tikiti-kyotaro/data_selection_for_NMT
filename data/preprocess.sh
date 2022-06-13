DATA=/home/hwichan/jpBART/data

echo 'binarize data'
fairseq-preprocess \
    --source-lang ko \
    --target-lang ja \
    --trainpref $DATA/train \
    --validpref $DATA/dev \
    --testpref  $DATA/test-n1 \
    --joined-dictionary \
    --destdir ko-ja