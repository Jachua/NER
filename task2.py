# import argparse
import random

TOKEN_INDEX = 0
POS_INDEX = 1
LABEL_INDEX = 2
SEPARATOR = "\t"


class LexiconBaselineSystem:
  def __init__(self, sentences):
    self.named_entities = set()
    self._setup(sentences)

  def _setup(self, sentences):
    for sentence in sentences:
      named_entity = None
      for token in sentence:
        if token[LABEL_INDEX].startswith("B"):
          if named_entity is not None:
            self.named_entities.add(named_entity)
          named_entity = token[TOKEN_INDEX]
        elif token[LABEL_INDEX].startswith("I"):
          assert(named_entity is not None)
          named_entity += SEPARATOR + token[TOKEN_INDEX]
        else:
          if named_entity is not None:
            self.named_entities.add(named_entity)
            named_entity = None

  def label(self, sentences):
    labels = []
    for sentence in sentences:
      label = list(map(lambda s: "O", sentence.split(SEPARATOR)))
      for entity in self.named_entities:
        index = sentence.find(entity)
        if index != -1 and sentence[index - 1] == SEPARATOR and ((index + len(entity)) == len(sentence) or sentence[index + len(entity)] == SEPARATOR):
          word_index = sentence[:index].count(SEPARATOR)
          label[word_index] = "B"
          for i in range(len(entity.split(SEPARATOR)) - 1):
            label[word_index + i] = "I"
      labels += label

    return labels


# Modified code from P1
class NGram(object):
  def __init__(self, sentences):
      self._build(sentences)

  def _build(self, sentences):
    self.bigrams = {}
    for sentence in sentences:
      for i in range(len(sentence) - 1):
        w1 = sentence[i]
        w2 = sentence[i + 1]
        if w1 not in self.bigrams:
          self.bigrams[w1] = {}
        if w2 not in self.bigrams[w1]:
          self.bigrams[w1][w2] = 0
        self.bigrams[w1][w2] += 1

  def prob(self, w1, w2):
    if w1 not in self.bigrams:
      return 0.0001
    elif w2 not in self.bigrams[w1]:
      return 0.0001
    # computes P(w2|w1)
    pw1 = self.bigrams[w1]
    pw1w2 = pw1[w2]
    count = sum(pw1.values())

    return pw1w2 / count


class HMMSystem:
  def __init__(self, sentences):
    self._setup(sentences)

  def _setup(self, sentences):
    pos = list(map(lambda sentence: list(map(lambda s: s[POS_INDEX], sentence)), sentences))
    self.states = set()
    for p in pos:
      self.states = self.states.union(set(p))
    self.states = list(self.states)

    self.transition_probabilities = NGram(pos)
    self.initial_probabilities = {}
    self.emission_probabilities = {}
    for state in self.states:
      self.initial_probabilities[state] = 0

    for sentence in sentences:
      for index in range(len(sentence)):
        s = sentence[index]
        token = s[TOKEN_INDEX]
        pos = s[POS_INDEX]
        if index == 0:
          self.initial_probabilities[pos] += 1

        if pos not in self.emission_probabilities:
          self.emission_probabilities[pos] = {}
        if token not in self.emission_probabilities[pos]:
          self.emission_probabilities[pos][token] = 0.0
        self.emission_probabilities[pos][token] += 1

    for pos in self.initial_probabilities:
      self.initial_probabilities[pos] = float(self.initial_probabilities[pos]) / len(sentences)

    for pos in self.emission_probabilities:
      count = sum(self.emission_probabilities[pos].values())
      for token in self.emission_probabilities[pos]:
        self.emission_probabilities[pos][token] /= count

  def prob(self, pos, token):
    pw1 = self.emission_probabilities[pos]
    pw1w2 = pw1[token]
    count = sum(pw1.values())

    return pw1w2 / count

  def viterbi(self, observations):
    num_observations = len(observations)
    num_states = len(self.states)

    viterbi = {}
    backpointer = {}

    first_observation = observations[0]
    for state in self.states:
      viterbi[state] = [0.0] * num_observations
      backpointer[state] = [0] * num_observations
      prob = self.emission_probabilities[state][first_observation] if first_observation in self.emission_probabilities[state] else 0.0001
      viterbi[state][0] = self.initial_probabilities[state] * prob

    max_terminating_score = 0
    max_terminating_state = None
    for t in range(1, num_observations):
      for s in self.states:
        for state in self.states:
          pobs = observations[t - 1]
          obs = observations[t]
          temp = viterbi[state][t - 1] * self.transition_probabilities.prob(pobs, obs)
          if obs in self.emission_probabilities[s]:
            temp *= self.emission_probabilities[s][obs]
          else:
            temp *= 0.0001

          if temp > viterbi[s][t]:
            backpointer[s][t] = state
            viterbi[s][t] = temp
            if t == num_observations - 1 and viterbi[s][t] > max_terminating_score:
              max_terminating_score = viterbi[s][t]
              max_terminating_state = s

    state = max_terminating_state
    labels = [state]
    for t in range(num_observations - 1, 0, -1):
      state = backpointer[state][t]
      labels = [state] + labels

    return labels

  def label(self, sentences):
    i = 0
    for sentence in sentences:
      print(sentence)
      print(self.viterbi(sentence.split(SEPARATOR)))
      i += 1
      if i == 3:
        raise Exception("")
#    return list(map(lambda sentence: self.viterbi(sentence.split(SEPARATOR)), sentences))


def check_system(system, sentences):
  tokens = []
  expected_labels = []
  for sentence in sentences:
    tokens.append(SEPARATOR.join(list(map(lambda s: s[TOKEN_INDEX], sentence))))
    for s in sentence:
      expected_labels.append(s[LABEL_INDEX].strip())

  actual_labels = system.label(tokens)
  print("Expected", len(expected_labels), "Actual", len(actual_labels))
  assert(len(expected_labels) == len(actual_labels))
  num_actual_labels = len(actual_labels)
  num_correct_labels = 0
  for index in range(num_actual_labels):
    if actual_labels[index] == expected_labels[index]:
      num_correct_labels += 1
  print("Precision:", float(num_correct_labels) * 100 / num_actual_labels)


def parse(name):
  f = open(name)
  lines = f.readlines()
  assert(len(lines) % 3 == 0)
  num_sentences = int(len(lines) / 3)
  sentences = []

  for i in range(num_sentences):
    sentence = list(zip(lines[3 * i].split(SEPARATOR), lines[3 * i + 1].split(SEPARATOR), lines[3 * i + 2].split(SEPARATOR)))
    sentences.append(sentence)

  return sentences


def main():
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
  sentences = parse("train.txt")
  num_validations = int(len(sentences) * 0.05)
  train_sentences = sentences[num_validations:]
  validation_sentences = sentences[:num_validations]

  baseline = LexiconBaselineSystem(train_sentences)
  check_system(baseline, validation_sentences)

  hmm = HMMSystem(train_sentences)
  check_system(hmm, validation_sentences)


if __name__ == '__main__':
  main()
