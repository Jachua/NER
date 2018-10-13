import math
import util

# import argparse
TOKEN_INDEX = 0
POS_INDEX = 1
LABEL_INDEX = 2
SEPARATOR = "\t"
EPSILON = math.log(0.001)
SUFFIX_LENGTH = 2


class LexiconBaselineSystem:
  def __init__(self, sentences):
    self.named_entities = set()
    self._setup(sentences)

  def _setup(self, sentences):
    for sentence in sentences:
      named_entity = None
      named_label = None
      for i in range(len(sentence[TOKEN_INDEX])):
        token = sentence[TOKEN_INDEX][i]
        label = sentence[LABEL_INDEX][i]

        if label.startswith("B"):
          if named_entity is not None:
            self.named_entities.add(named_entity + "---" + named_label)
          named_entity = token
          named_label = label.split("-")[1]
        elif label.startswith("I"):
          assert(named_entity is not None)
          named_entity += SEPARATOR + token
        else:
          if named_entity is not None:
            self.named_entities.add(named_entity + "---" + named_label)
            named_entity = None
            named_label = None

  def label(self, sentences):
    labels = []
    for tokens in sentences:
      sentence = SEPARATOR.join(tokens)
      label = list(map(lambda s: "O", tokens))
      for entity in self.named_entities:
        [name, l] = entity.split("---")
        index = sentence.find(name)
        if index != -1 and sentence[index - 1] == SEPARATOR and ((index + len(name)) == len(sentence) or sentence[index + len(name)] == SEPARATOR):
          word_index = sentence[:index].count(SEPARATOR)
          label[word_index] = "B-" + l
          for i in range(len(name.split(SEPARATOR)) - 1):
            label[word_index + i] = "I-" + l
      labels += label

    return labels


# Modified code from P1
class NGram(object):
  def __init__(self, sentences):
      self._build(sentences)

  def _build(self, sentences):
    self.bigrams = {}
    self.tags = set()
    for sentence in sentences:
      for i in range(len(sentence) - 1):
        w1 = sentence[i]
        w2 = sentence[i + 1]
        self.tags.add(w1)
        self.tags.add(w2)
        if w1 not in self.bigrams:
          self.bigrams[w1] = {}
        if w2 not in self.bigrams[w1]:
          self.bigrams[w1][w2] = 0
        self.bigrams[w1][w2] += 1

    for tag1 in self.tags:
      for tag2 in self.tags:
        if tag1 not in self.bigrams:
          self.bigrams[tag1] = {}
        if tag2 not in self.bigrams[tag1]:
          self.bigrams[tag1][tag2] = 0

        if not ((tag1.startswith("B") or tag1.startswith("I")) and tag2.startswith("I") and tag1.split("-")[1] != tag2.split("-")[1]):
          self.bigrams[tag1][tag2] += 1

  def prob(self, w1, w2):
    pw1 = self.bigrams[w1]
    pw1w2 = pw1[w2]
    count = sum(pw1.values())

    return pw1w2 / count


class HMMSystem:
  def __init__(self, sentences):
    self._setup(sentences)

  def normalize(self, probabilities):
    for label in probabilities:
      count = sum(probabilities[label].values())
      for token in probabilities[label]:
        probabilities[label][token] /= count
        probabilities[label][token] = math.log(probabilities[label][token])

  def smooth(self, probabilities, tokens):
    for label in probabilities:
      for token in tokens:
        if token not in probabilities[label]:
          probabilities[label][token] = 0
        probabilities[label][token] += 1

  def _setup(self, sentences):
    label = list(map(lambda sentence: sentence[LABEL_INDEX], sentences))
    self.states = set()
    for l in label:
      self.states = self.states.union(set(l))
    self.states = list(self.states)
    # Made order we look at states deterministic
    self.states.sort()

    self.transition_probabilities = NGram(label)
    self.initial_probabilities = {}
    self.emission_probabilities = {}
    self.backoff_probabilities = {}
    for state in self.states:
      self.initial_probabilities[state] = 0

    suffixes = set()
    tokens = set()
    for sentence in sentences:
      for index in range(len(sentence[TOKEN_INDEX])):
        token = sentence[TOKEN_INDEX][index]
        label = sentence[LABEL_INDEX][index]
        tokens.add(token)
        if index == 0:
          self.initial_probabilities[label] += 1

        if label not in self.emission_probabilities:
          self.emission_probabilities[label] = {}
        if token not in self.emission_probabilities[label]:
          self.emission_probabilities[label][token] = 0.0

        # Suffix backoff. Many POS have same morphology.
        # For example, nouns tend to end in "-s" and verbs tend to end in "-ed"
        # For cases where we can't find a word, we will attempt to use the suffix
        # to assign a probability
        for suffix_length in range(1, SUFFIX_LENGTH + 1):
          suffix = token[-suffix_length:]
          if len(suffix) > 0:
            if label not in self.backoff_probabilities:
              self.backoff_probabilities[label] = {}
            if suffix not in self.backoff_probabilities[label]:
              self.backoff_probabilities[label][suffix] = 0
              suffixes.add(suffix)
            self.backoff_probabilities[label][suffix] += 1

        self.emission_probabilities[label][token] += 1

    # Initial probabilites for labels
    for label in self.initial_probabilities:
      if self.initial_probabilities[label] == 0:
        self.initial_probabilities[label] = float("-inf")
      else:
        self.initial_probabilities[label] = math.log(float(self.initial_probabilities[label]) / len(sentences))

    # Plus one smoothing for emissions and backoff probabilities
    suffixes.add("")
    self.smooth(self.backoff_probabilities, suffixes)
    self.smooth(self.emission_probabilities, tokens)

    # Normalize probabilities
    self.normalize(self.emission_probabilities)
    self.normalize(self.backoff_probabilities)

  def prob(self, label, token):
    pw1 = self.emission_probabilities[label]
    pw1w2 = pw1[token]
    count = sum(pw1.values())

    return pw1w2 / count

  def viterbi(self, observations):
    observations = list(map(lambda o: o.strip(), observations))
    num_observations = len(observations)
    num_states = len(self.states)

    viterbi = {}
    backpointer = {}

    first_observation = observations[0]
    for state in self.states:
      viterbi[state] = [None] * num_observations
      backpointer[state] = [float("-inf")] * num_observations
      prob = self.emission_probabilities[state][first_observation] if first_observation in self.emission_probabilities[state] else EPSILON
      viterbi[state][0] = self.initial_probabilities[state] + prob

    max_terminating_score = float("-inf")
    max_terminating_state = None
    for t in range(1, num_observations):
      for prev_state in self.states:
        for curr_state in self.states:
          pobs = observations[t - 1]
          obs = observations[t]
          temp = viterbi[curr_state][t - 1] + self.transition_probabilities.prob(prev_state, curr_state)
          if obs in self.emission_probabilities[curr_state]:
            temp += self.emission_probabilities[curr_state][obs]
          else:
            for suffix_length in range(SUFFIX_LENGTH, -1, -1):
              assert(suffix_length >= 0)
              suffix = obs[-suffix_length:]
              if suffix in self.backoff_probabilities[curr_state]:
                assert(self.backoff_probabilities[curr_state][suffix] != 0)
                temp += self.backoff_probabilities[curr_state][suffix]
                break
              if suffix_length == 0:
                temp += EPSILON

          if viterbi[curr_state][t] is None or (temp > viterbi[curr_state][t]) or (temp == viterbi[curr_state][t] and prev_state == "O"):
            backpointer[curr_state][t] = prev_state
            viterbi[curr_state][t] = temp
            if t == num_observations - 1:
              if viterbi[curr_state][t] > max_terminating_score:
                max_terminating_score = viterbi[curr_state][t]
                max_terminating_state = curr_state

    state = max_terminating_state
    labels = [state]
    for t in range(num_observations - 1, 0, -1):
      state = backpointer[state][t]
      labels = [state] + labels

    return labels

  def label(self, sentences):
    labels = []
    for sentence in sentences:
      labels += self.viterbi(sentence)
    return labels


def check_system(system, sentences):
  tokens = []
  expected_labels = []
  for sentence in sentences:
    tokens.append(sentence[TOKEN_INDEX])
    expected_labels += sentence[LABEL_INDEX]

  actual_labels = system.label(tokens)
  print("Expected", len(expected_labels), "Actual", len(actual_labels))
  assert(len(expected_labels) == len(actual_labels))
  num_correct_labels = 0
  num_actual_labels = 0
  for index in range(len(actual_labels)):
    if actual_labels[index] != "O":
      num_actual_labels += 1
      if actual_labels[index] == expected_labels[index]:
        num_correct_labels += 1
  print("Precision:", float(num_correct_labels) * 100 / num_actual_labels)
  num_correct_labels = 0
  num_expected_labels = 0
  for index in range(len(expected_labels)):
    if expected_labels[index] != "O":
      num_expected_labels += 1
      if actual_labels[index] == expected_labels[index]:
        num_correct_labels += 1
  print("Recall:", float(num_correct_labels) * 100 / num_expected_labels)


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
  train_set, dev_set = util.preprocess("train.txt", is_train=False)
  print("Baseline")
  baseline = LexiconBaselineSystem(train_set)
  check_system(baseline, dev_set)

  print("HMM")
  hmm = HMMSystem(train_set)
  check_system(hmm, dev_set)


if __name__ == '__main__':
  main()
