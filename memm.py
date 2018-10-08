import random
import numpy as np
from nltk.classify import maxent

import argparse

START = '</s>'
SSTART = '<//s>'
END = '<s/>'
EEND = '<s//>'

def preprocess(train_file):
    with open(train_file) as f:
        raw = f.read().split('\n')
        f.close()
    size = len(raw)//3
    data = []
    for i in range(0, len(raw), 3):
        data.append([raw[i].split(), raw[i + 1].split(), raw[i + 2].split()])

    dev_idx = random.sample(range(size), size//5)
    train_idx = np.setdiff1d(range(size), dev_idx)
    train_set = [data[idx] for idx in train_idx]
    dev_set = [data[idx] for idx in dev_idx]
    return train_set, dev_set

def update(token, bank, next_idx):
    if not token in bank:
        bank[token] = next_idx
        next_idx += 1
    return next_idx


class MEMM(object):

    def __init__(self, data):
        word_bank = {SSTART: 0, START: 1, END: 2, EEND: 3}
        pos_bank = {SSTART: 0, START: 1, END: 2, EEND: 3}
        ner_bank = {SSTART: 0, START: 1, END: 2, EEND: 3}
        self.bank = [word_bank, pos_bank, ner_bank]
        # [next_word_idx, next_pos_idx, next_ner_idx]
        self.next_idx = [4, 4, 4]

        self.train = []
        self.dev = []
        self.test = []

        self._from_data_train(data)

    # token = [word, POS, NER]
    def set_token(self, sample, idx, size, is_bigram = False):
        if not is_bigram:
            bound = [-2, -1, size, size + 1]
            for i in range(4):
                if idx == bound[i]:
                    return np.full(3, i)
            token = sample[:, idx]

        if is_bigram:
            token = []
            if idx == -1:
                for i in range(3):
                    token.append(START + ' ' + sample[i, 0])
            if idx == size - 1:
                for i in range(3):
                    token.append(sample[i, size - 1] + ' ' + END)
            else:
                for i in range(3):
                    token.append(sample[i, idx] + ' ' + sample[i, idx + 1])

        token_idx = np.zeros(3)
        for i in range(3):
            self.next_idx[i] = update(token[i], self.bank[i], self.next_idx[i])
            token_idx[i] = self.bank[i][token[i]]
        return token_idx

    # sample = [[word, ..., word], 
    #           [POS, ..., POS], 
    #           [NER, ..., NER]]
    #           np.array
    def construct_feature(self, sample, data_set):
        size = len(sample[0])
        for i in range(size):
            # np.array([word_idx, POS_idx, NER_idx])
            pprev = self.set_token(sample, i - 2, size)
            prev = self.set_token(sample, i - 1, size)
            post = self.set_token(sample, i + 1, size)
            ppost = self.set_token(sample, i + 2, size)

            prev_bigram = self.set_token(sample, i - 1, size, is_bigram=True)
            post_bigram = self.set_token(sample, i, size, is_bigram=True)

            # [w, pos]
            cur_w_pos = []
            for j in range(2):
                self.next_idx[j] = update(sample[j, i], self.bank[j], self.next_idx[j])
                cur_w_pos.append(self.bank[j][sample[j, i]])

            d = {
                'w_pprev': pprev[0], 'w_prev': prev[0], 'w': cur_w_pos[0], 'w_post': post[0],
                'w_ppost': ppost[0], 'w_prev_bigram': prev_bigram[0], 'w_post_bigram': post_bigram[0],
                'pos_pprev': pprev[1], 'pos_prev': prev[1], 'pos': cur_w_pos[1], 'pos_post': post[1],
                'pos_ppost': ppost[1], 'pos_prev_bigram': prev_bigram[1], 'pos_post_bigram': post_bigram[1],
                'ner_pprev': pprev[2], 'ner_prev': prev[2], 'ner_prev_bigram': prev_bigram[2]
                }
            data_set.append((d, sample[2, i]))
    
    def _from_data_train(self, data):
        for sample in data:
            self.construct_feature(np.array(sample), self.train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', dest = 'train_file', default = 'sample.txt')
    args = parser.parse_args()
    train_set, dev_set = preprocess(args.train_file)
    model = MEMM(train_set)