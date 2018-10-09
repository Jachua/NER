# import argparse
import random

TOKEN_INDEX = 0
POS_INDEX = 1
LABEL_INDEX = 2
SEPARATOR = "\t"

class LexiconBaselineSystem:
  def __init__(self, sentences):
    self.named_entities = set()
    self.setup(sentences)

  def setup(self, sentences):
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


if __name__ == '__main__':
  main()
