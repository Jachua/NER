# Named Entity Recognition 

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