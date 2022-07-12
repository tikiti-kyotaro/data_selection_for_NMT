GPU1=$1
GPU2=$2
CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 fairseq-generate /home/kyotaro/data_selection_for_NMT/data/preprocesses/preprocess-full-final2/en-de-full-remake-final2 \
    --path $models /home/kyotaro/data_selection_for_NMT/checkpoints/checkpoint-full-final2/checkpoint-full-final2/checkpoint_best.pt \
    --task translation \
    --gen-subset test \
    --batch-size 128 \
    --batch-size 128 --beam 5 | tee output.data_selection.nbest.txt

grep "^H" output.data_selection.nbest.txt | LC_ALL=C sort -V | cut -f3- > output.data_selection.nbest.data.txt

# 翻訳結果だけを出力
cat output.data_selection.nbest.data.txt | sed -r 's/(@@ )|(@@ ?$)//g' > output.result.txt

# detoknizer
/home/kyotaro/data_selection_for_NMT/tools/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de < output.result.txt > data_selection_test.txt

# 元のデータのデトークナイズ
/home/kyotaro/data_selection_for_NMT/tools/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de < /home/kyotaro/data_selection_for_NMT/data/wmt17_en_de/tmp/test.de > /home/kyotaro/data_selection_for_NMT/data/wmt17_en_de/ref/test.de

# BLUE
cat data_selection_test.txt | sacrebleu -w 2 /home/kyotaro/data_selection_for_NMT/valid_test/test_unbpe_final.de

# BERTScore
# cat data_selection_test.txt | bert-score -w 2 /home/kyotaro/data_selection_for_NMT/data/wmt17_en_de/ref/test.de