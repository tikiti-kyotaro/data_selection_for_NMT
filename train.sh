GPU=$1
BIN=/home/kyotaro/data_selection_for_NMT/data/preprocess-fda/en-de-900k-fda

CUDA_VISIBLE_DEVICES=$GPU fairseq-train $BIN \
    --seed 1 \
    --keep-last-epochs 1 \
    --arch transformer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0005 \
    --update-freq 8 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-update 50000 \
    --source-lang en \
    --target-lang de \
    --patience 10 \
    --save-dir checkpoint-fda-900k > train.log