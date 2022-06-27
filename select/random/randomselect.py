import random

def randomselect(en_input, de_input, en_output, de_output, N):
    with open(en_input, "r") as en_input, open(de_input, "r") as de_input:
        with open(en_output, "w") as en_out, open(de_output, "w") as de_out:
            parallel = []
            for en_line, de_line in zip(en_input, de_input):
                parallel.append((en_line, de_line))

            random_parallel = random.sample(parallel, N)

            for i in range(len(random_parallel)):
                en_out.write(random_parallel[i][0])
                de_out.write(random_parallel[i][1])

random.seed(0)
randomselect("/home/kyotaro/data_selection_for_NMT/data/wmt17_en_de/train.en", \
"/home/kyotaro/data_selection_for_NMT/data/wmt17_en_de/train.de", \
"/home/kyotaro/data_selection_for_NMT/data/random-train/500k/train.random.en", \
"/home/kyotaro/data_selection_for_NMT/data/random-train/500k/train.random.de", 500000)