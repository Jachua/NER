import csv
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

def csv_out(preds, indices):
    prev_idx = 0
    prev_tag = preds[0][0][-4:]
    output = {'-PER': [], '-LOC': [], '-ORG': [], 'MISC': []}
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            cur_idx = indices[i][j]
            cur_tag = preds[i][j][-4:]
            if cur_tag != prev_tag:
                if prev_tag != 'O':
                    output[prev_tag[-4:]].append(str(prev_idx) + '-' + str(cur_idx - 1))
                prev_idx = cur_idx
                prev_tag = cur_tag
    if prev_tag != 'O':
        output[prev_tag[-4:]].append(str(prev_idx) + '-' + str(cur_idx - 1))
    type_arr = ['PER', 'LOC', 'ORG', 'MISC']
    pred_arr = [' '.join(output['-PER']), ' '.join(output['-LOC']), ' '.join(output['-ORG']),
                ' '.join(output['MISC'])]
    with open('memm.csv', 'w+') as csvfile:
      w = csv.writer(csvfile, delimiter=',')
      w.writerow(['Type', 'Prediction'])
      for i in range(4):
        w.writerow([type_arr[i], pred_arr[i]])

