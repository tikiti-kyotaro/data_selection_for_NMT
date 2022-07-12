from bert_score import score

def calc_bert_score(cands, refs):
    """ BERTスコアの算出

    Args:
        cands ([List[str]]): [比較元の文]
        refs ([List[str]]): [比較対象の文]

    Returns:
        [(List[float], List[float], List[float])]: [(Precision, Recall, F1スコア)]
    """
    Precision, Recall, F1 = score(cands, refs, lang="de", verbose=True)
    return Precision.numpy().tolist(), Recall.numpy().tolist(), F1.numpy().tolist()


if __name__ == "__main__":
    """ サンプル実行 """ 
    
    p_result = 0
    r_result = 0
    f1_result = 0

    with open("/home/kyotaro/data_selection_for_NMT/eval/data_selection_test.txt") as f:
        cands = [line.strip() for line in f]

    with open("/home/kyotaro/data_selection_for_NMT/valid_test/test_unbpe_final.de") as f:
        refs = [line.strip() for line in f]
    
    P, R, F1 = calc_bert_score(cands, refs)
    for p,r, f1 in zip(P, R, F1):
        #print("P:%f, R:%f, F1:%f" %(p, r, f1))
        p_result += p
        r_result += r
        f1_result += f1

    print(f'Preccission : {p_result / len(P)}')
    print(f'Recall : {r_result / len(R)}')
    print(f'F1 : {f1_result / len(F1)}')