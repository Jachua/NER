import numpy as np
import random

def preprocess(train_file, is_test=False, is_train=False):
    random.seed(0)
    with open(train_file) as f:
        raw = f.read().split('\n')
        f.close()
    size = len(raw)//3
    data = []
    if is_test:
        indices = []
        for i in range(0, len(raw), 3):
            word_arr = list(map(lambda r: r.lower(), raw[i].split()))
            line_length = len(word_arr)
            default_ner = np.full(line_length, 'O', dtype="<U10")
            data.append([word_arr, raw[i + 1].split(), default_ner])
            indices.append([int(idx) for idx in raw[i + 2].split()])
        return data, indices

    for i in range(0, len(raw), 3):
        data.append([list(map(lambda r: r.lower(), raw[i].split())), raw[i + 1].split(), raw[i + 2].split()])

    # data split into 80% training, 20% validation
    if is_train:
        return data, []
    dev_idx = random.sample(range(size), size//5)
    train_idx = np.setdiff1d(range(size), dev_idx)
    train_set = [data[idx] for idx in train_idx]
    dev_set = [data[idx] for idx in dev_idx]
    return train_set, dev_set
