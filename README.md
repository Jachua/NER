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


Could take up to hours to train the MaxEnt Classifier from the NLTK package. Converges after ~20 iterations for the training set (i.e. 80\% of train.txt) when log likelihood = -0.035. 


Most informative features generated from the MaxEnt classifier ranked by weights include:
* whether the letters in the target word are all upper case (most significant for I-ORG, I-LOC, I-MISC)
* POS tags for words around the target word (most significant for B-MISC, I-PER)
* whether the target word contains hyphen (most significant for I-ORG, I-LOC), contain number (most significant for I-ORG, I-MISC)
* shape of the target word (most significant for O)

All weights associated with the above features are negative, suggesting that the classifier relies strongly on elimination. Curiously, the NER tag for the previous word does not factor heavily into the classification, so viterbi contributes less to the predictions for MEMM than in HMM. Rather, the most informative features are derived from information about word shape.