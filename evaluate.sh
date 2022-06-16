GPU=$1
MODEL=$2 # path to model
BIN_DATA=$3 # path to binalized data
ROOT=.
MOSES=path/to/moses
REF=path/to/reference
SRC=en
TGT=de

mkdir -p outputs
mkdir -p outputs/$MODEL

# Translate
for split in test dev; do
    output_file=outputs/$MODEL/$split.out.$TGT
    bpe_file=outputs/$MODEL/$split.bpe.$TGT
    tok_file=outputs/$MODEL/$split.tok.$TGT
    final_file=outputs/$MODEL/$split.$TGT

    if [ ! -f ${output_file} ]; then
       CUDA_VISIBLE_DEVICES=$GPU PYTHONIOENCODING=utf-8 fairseq-generate $BIN_DATA \
            --path $MODEL \
            --gen-subset $split \
            -s $SRC -t $TGT \
            --scoring sacrebleu > $sp_file
    fi

    # Reordering
    if [ ! -f ${bpe_file} ]; then
        # reordering 処理
    fi

    # Debpe
    if [ ! -f ${tok_file} ]; then
        cat ${bpe_file} | sed -r 's/(@@ )|(@@ ?$)//g' > ${tok_file}
    fi

    # De-tokenizing
    if [ ! -f ${final_file} ]; then
       ${MOSES}/tokenizer/detokenizer.perl -l $TGT < ${tok_file} > ${final_file}
    fi

    # Evaluate using sacreBLEU (WMT official scorer)
    cat ${final_file} | sacrebleu -w 2 $REF/${split}.$TGT

done
done
done
