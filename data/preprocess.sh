DATA=/home/kyotaro/data_selection_for_NMT/data/wmt17_en_de

echo 'binarize data'
fairseq-preprocess \
    --source-lang en \
    --target-lang de \
    --trainpref /home/kyotaro/data_selection_for_NMT/data/train-random/700k/train.random \
    --validpref $DATA/valid \
    --testpref  $DATA/test \
    --joined-dictionary \
    --destdir preprocess-random/en-de-700k-random