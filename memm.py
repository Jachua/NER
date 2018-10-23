import random
import numpy as np
from nltk.classify import maxent
import re
import argparse
import csv
import util

START = '</s>'
SSTART = '<//s>'
END = '<s/>'
EEND = '<s//>'


def update(token, bank, next_idx):
    if not token in bank:
        bank[token] = next_idx
        next_idx += 1
    return next_idx

def wordshape(text, is_short=False):
    sp = ''
    if is_short:
        sp = '+'
    replace_upper = re.sub('[A-Z]' + sp, 'X', text)
    replace_lower = re.sub('[a-z]' + sp, 'x', replace_upper)
    shape = re.sub('[0-9]' + sp, 'd', replace_lower)
    return shape

class MEMM(object):

    def __init__(self, data):
        word_bank = {SSTART: 0, START: 1, END: 2, EEND: 3}
        pos_bank = {SSTART: 0, START: 1, END: 2, EEND: 3}
        ner_bank = {SSTART: 0, START: 1, END: 2, EEND: 3}
        self.bank = [word_bank, pos_bank, ner_bank]
        # [next_word_idx, next_pos_idx, next_ner_idx]
        self.next_idx = [4, 4, 4]

        self.w_pos_bank = {}
        self.w_pos_next_idx = 0

        self.tags = set()
        # number of possible tags
        self.state_size = 0
        # map indices to tags
        self.idx2state = {}

        self.train = []
        self.dev = []
        self.test = []

        self.shape_bank = {}
        self.shape_next_idx = 0

        self._from_data_train(data)

    # token = [word, POS, NER]
    def set_token(self, sample, idx, size, is_bigram=False):
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
    def construct_feature(self, sample, data_set, is_test=False):
        if is_test:
            line_feature = []
        size = len(sample[0])
        for i in range(size):
            # np.array([word_idx, POS_idx, NER_idx])
            pprev = self.set_token(sample, i - 2, size)
            prev = self.set_token(sample, i - 1, size)
            post = self.set_token(sample, i + 1, size)
            ppost = self.set_token(sample, i + 2, size)

            prev_bigram = self.set_token(sample, i - 1, size, is_bigram=True)
            post_bigram = self.set_token(sample, i, size, is_bigram=True)

            # w_pos_idx
            w_pos = sample[0, i] + ' ' + sample[1, i]
            self.w_pos_next_idx = update(w_pos, self.w_pos_bank, self.w_pos_next_idx)
            w_pos_idx = self.w_pos_bank[w_pos]

            # [w_idx, pos_idx]
            cur_w_pos = []
            for j in range(2):
                self.next_idx[j] = update(sample[j, i], self.bank[j], self.next_idx[j])
                cur_w_pos.append(self.bank[j][sample[j, i]])

            # features for unknown words
            w = sample[0, i]
            shape = wordshape(w)
            shape_short = wordshape(w, True)
            self.shape_next_idx = update(shape, self.shape_bank, self.shape_next_idx)
            self.shape_next_idx = update(shape_short, self.shape_bank, self.shape_next_idx)
            shape_idx = self.shape_bank[shape]
            shape_short_idx = self.shape_bank[shape_short]
            contain_num = int('d' in shape_short)
            contain_upper = int('X' in shape_short)
            contain_hyphen = int('-' in shape_short)
            all_upper = int(w.isupper())

            # TODO: improve ner_prev with B-NER and I-NER
            d = {
                'w_pprev': pprev[0], 'w_prev': prev[0], 'w': cur_w_pos[0], 'w_post': post[0],
                'w_ppost': ppost[0], 'w_prev_bigram': prev_bigram[0], 'w_post_bigram': post_bigram[0],
                'pos_pprev': pprev[1], 'pos_prev': prev[1], 'pos': cur_w_pos[1], 'pos_post': post[1],
                'pos_ppost': ppost[1], 'pos_prev_bigram': prev_bigram[1], 'pos_post_bigram': post_bigram[1],
                'w_pos': w_pos_idx,'ner_prev': prev[2], 'shape': shape_idx, 'shape_short': shape_short_idx,
                'contain_num': contain_num, 'contain_upper': contain_upper, 'contain_hyphen': contain_hyphen,
                'all_upper': all_upper
                # 'ner_pprev': pprev[2], 'ner_prev_bigram': prev_bigram[2]
                }
            if is_test:
                line_feature.append(d)
            else:
                data_set.append((d, sample[2, i]))
                self.tags.add(sample[2, i])

        if is_test:
            data_set.append(line_feature)
        else:
            self.state_size = len(self.tags)
            self.idx2state = dict(zip(range(self.state_size), self.tags))

    def _from_data_train(self, data):
        for sample in data:
            self.construct_feature(np.array(sample), self.train)
        self.classifier = maxent.MaxentClassifier.train(self.train, trace=5)
        self.classifier.show_most_informative_features(n=30)

    # data = [[word, ..., word],
    #         [POS, ..., POS]]
    def from_data_test(self, data):
        for sample in data:
            # featuresets separated by lines
            self.construct_feature(np.array(sample), self.test, is_test=True)
        preds = []
        for line in self.test:
            preds.append(self.viterbi(line))
        return preds

    def prev_max(self, v_prev, ft, ner):
        s = np.zeros(self.state_size)
        for i in range(self.state_size):
            ner_prev = self.idx2state[i]
            ft['ner_prev'] = self.bank[2][ner_prev]
            prob_d = self.classifier.prob_classify(ft)
            s[i] = prob_d.prob(ner)*v_prev[i]
        max_idx = np.argmax(s)
        # maximum value  and index given current NER over previous NER
        return s[max_idx], max_idx

    def recover_path(self, bp, bestpathpointer, ob_size):
        path = np.zeros(ob_size, dtype=int)
        path[-1] = bestpathpointer
        for i in range(ob_size - 2, -1, -1):
            path[i] = bp[i + 1, path[i + 1]]
        return [self.idx2state[idx] for idx in path]

    # featuresets for a line in test
    def viterbi(self, line):
        state_size = self.state_size
        idx2state = self.idx2state
        ob_size = len(line)

        #   [[...]
        # o  [...]
        # b  [...]
        #    [...]]
        #    state
        v = np.zeros((ob_size, state_size))
        bp = np.zeros((ob_size, state_size), dtype=int)

        prob_init = self.classifier.prob_classify(line[0])
        for i in range(state_size):
            v[0, i] = prob_init.prob(idx2state[i])

        for i in range(1, ob_size):
            ft = line[i]
            for j in range(state_size):
                ner = idx2state[j]
                v[i, j], bp[i, j] = self.prev_max(v[i - 1, :], ft, ner)

        bestpathpointer = np.argmax(v[-1, :])
        bestpathprob = v[-1, bestpathpointer]
        print("best path probability ", bestpathprob)
        return self.recover_path(bp, bestpathpointer, ob_size)

# def preprocess_test(test_file):
#     with open(test_file) as f:
#         raw = f.read().split('\n')
#         f.close()
#     data = []

#     for i in range(0, len(raw), 3):
#         data.append([raw[i].split(), raw[i + 1].split(), raw[i + 2].split()])

#     return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', dest = 'train_file', default = 'sample.txt')
    parser.add_argument('--test_file', dest = 'test_file', default = 'sample.txt')
    args = parser.parse_args()


    # train_set, dev_set = preprocess(args.train_file)
    train_set, _ = util.preprocess(args.train_file, is_train=True)
    model = MEMM(train_set)

    # preds = model.from_data_test(dev_set)
    # for i in range(len(preds)):
    #     print("\n\n=====\n\nPredictions\n", preds[i])
    #     print("\n\n=====\n\nCorrect tags\n", dev_set[i][2])

    test_set, indices = util.preprocess(args.test_file, is_test=True)
    # test_set = preprocess_test(args.test_file)
    # model = MEMM(test_set)

    preds = model.from_data_test(test_set)
    util.csv_out(preds, indices)
