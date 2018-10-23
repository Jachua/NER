# Named Entity Recognition
## HMM

To run HMM, run ```python3 task2.py```. This will run HMM on the test dataset and print out the precision/recall values.

## Maximum Entropy Markov Model
Feature template

$
<NER_i, w_{i - 2}>, <NER_i, w_{i - 1}>, <NER_i, w_i, w_{i + 1}>, <NER_i, w_{i + 2}>, <NER_i, w_{i - 1}, w_i>, <NER_i, w_iw_{i + 1}>, <NER_i, POS_{i - 2}>, <NER_i, POS_{i - 1}>, <NER_i, POS_i>, <NER_i, w_{i + 1}>, <NER_i, POS_{i + 2}>, <NER_i, POS_{i - 1}, POS_i>, <NER_i, POS_i, POS_{i + 1}> <NER_i, NER_{i - 1}>
$

Specifically, features for unknown words

$
<NER_i, \textbf{word shape}> <NER_i, \textbf{word shape shortened}> <NER_i, \textbf{word contains numbers}> <NER_i, \textbf{word contains upper case letters}> <NER_i, \textbf{word contains hyphens}> <NER_i, \textbf{all letters in word are upper case}>
$

Trained with maximum entropy classifier from [NLTK](https://github.com/nltk/nltk) package. Decoded with Viterbi algorithm.


Most informative features generated from the MaxEnt classifier ranked by weights include:
* whether $w_i$ contains uppercase letters - negative weights when the word does not contain uppercase letters, most significant for B-LOC, B-MISC, B-PER, B-ORG
* $w_{i - 1}, w_i$ - positive weights, most significant for B-LOC, I-ORG, I-MISC
* $w_{i - 1}$ - positive weights, most significant for I-LOC
* $w_{i + 1}$ - positive weights, most significant for B-MISC, B-LOC
* $w_i, w_{i + 1}$ - positive weights, most significant for B-MISC, I_MISC, O
* $NER_{i - 1}$ - negative weights, most significant for B-PER, B-LOC, B-PER
* $POS_i$ - negative weights, most significant for B-LOC
* word shape - negative weights, most significant for I-ORG
* $POS_{i + 1}$ - negative weights, most significant for I-PER
