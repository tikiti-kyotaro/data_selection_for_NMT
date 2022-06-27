from collections import defaultdict
import math
from numpy import source
from tqdm import tqdm


class FDA:
    def __init__(self, N):
        self.features = defaultdict(lambda: 0)  # 特徴量とその個数
        self.fvalue = defaultdict(lambda: 0)  # 特徴量のスコア
        self.L_size = 0
        self.L = list()  # 最終的なスコア
        self.Q = list()  # 優先度付きキュー
        self.score = defaultdict(lambda: 0)  # 各文のスコア
        self.cost = N  # アノテーションの許容量
        self.sentence_pairs = list()  # 対訳データのペア
        self.sentences_score = defaultdict(lambda: 0)

    def selection(self, sorce_data, target_data):
        """文選択"""
        with open(sorce_data, "r") as data, open(target_data, "r") as result:
            self.setting(data, result)
            self.first_loop()  # ok
            self.second_loop()
            self.final_loop()
    
### 前準備 ###
    def setting(self, data, result):
        """特徴量、文書サイズ、対訳文を設定"""
        for sorce, target in tqdm(zip(data, result)):
            sorce = sorce.strip()
            target = target.strip()
            self.sentence_pairs.append((sorce, target))
            words = sorce.split()
            pre_features = self.make_features(words)
            for feature in pre_features:
                self.features[feature] += 1
            self.L_size += 1

    def make_features(self, words):
        """featureの作成"""
        result = list()
        self.L_size = len(words)
        for i in range(len(words) - 1):
            result.append(" ".join(words[i:i+2]))
        return result

### 最初のループ ###
    def first_loop(self):
        """最初のループ"""
        for feature, count in tqdm(self.features.items()):
            self.fvalue[feature] = self.init(feature)
    
    def init(self, feature):
        """特徴量のスコアの初期化"""
        return math.log(self.L_size / self.features[feature])


### ２番目のループ ###
    def second_loop(self):
        """２番目のループ"""
        for sentence_pair in tqdm(self.sentence_pairs):
            sorce = sentence_pair[0]
            self.cal_score(sorce)
            self.push(sentence_pair)

    def cal_score(self, sentence):
        """スコアの計算"""
        words = "".join(sentence).strip().split()
        sentence_features = self.make_features(words)
        for feature in sentence_features:
            self.score[" ".join(words)] += self.fvalue[feature]

    def push(self, sentence_pair):
        """Qに要素を追加して、ソート（一文づつ追加）"""
        if len(self.Q) == 0:
            self.Q.append(sentence_pair)
        else:
            for i in range(len(self.Q)):
                if self.score[sentence_pair[0]] > self.score[self.Q[i][0]]:
                    self.Q.insert(i, sentence_pair)
                    break
                self.Q.append(sentence_pair)
                break

### 最後のループ ###
    def final_loop(self):
        """最後のループ"""
        with tqdm() as pbar:
            while len(self.L) < self.cost:
                S = self.Q.pop(0)
                sorce = S[0]
                self.cal_score(sorce)
                if self.score[sorce] >= self.topval():
                    self.L.append(S)  # ok
                    sorce_words = sorce.split()
                    sorce_features = self.make_features(sorce_words)
                    for feature in sorce_features:
                        self.decay(feature)
                else:
                    self.push(S)
                pbar.update(1)

    def cal_score_final(self, sentence):
        """スコアの計算"""
        words = "".join(sentence).strip().split()
        sentence_features = self.make_features(words)
        for feature in sentence_features:
            self.score[sentence] += self.fvalue[feature]

    def topval(self):
        """Qに入っている対訳データのソース言語文のスコア"""
        return self.score[self.Q[0][0]]

    def decay(self, feature):
        """FDA"""
        self.fvalue[feature] = self.init(feature) / 1 + self.features[feature]

    def out_put(self, sorce_out, target_out):
        """出力"""
        with open(sorce_out, "w") as sorce, open(target_out, "w") as target:
            for pair in self.L:
                sorce.write(pair[0])
                sorce.write("\n")
                target.write(pair[1])
                target.write("\n")


##########################################################


    def test_sentence_pairs(self):
        for i in range(len(self.sentence_pairs)):
            print(self.sentence_pairs[i])

    def test_features(self):
        for feature in self.features:
            print(feature)

    def test_fvalue(self):
        for key, value in self.fvalue.items():
            print(f'{key} : {value}')

    def test_score(self):
        for key, value in self.score.items():
            print(f'{key} : {value}')

    def test_Q(self):
        for i in range(len(self.Q)):
            print(self.Q[i][0])

    def test_L(self):
        for sentence in self.L:
            print(sentence)

###########################################################



if __name__ == "__main__":
    sorce_data = "/home/kyotaro/data_selection_for_NMT/data/wmt17_en_de/train.en"
    target_data = "/home/kyotaro/data_selection_for_NMT/data/wmt17_en_de/train.de"
    sorce_out = "/home/kyotaro/data_selection_for_NMT/data/random-fda/1m/train.fda.en"
    target_out = "/home/kyotaro/data_selection_for_NMT/data/random-fda/1m/train.fda.de"
    fda = FDA(1000000)
    fda.selection(sorce_data, target_data)
    # fda.test_sentence_pairs()
    # fda.test_features()
    # fda.test_fvalue()
    # fda.test_score()
    # fda.test_Q()
    # fda.test_L()
    fda.out_put(sorce_out, target_out)