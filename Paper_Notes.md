# Transformer Architecture Overview

## Central Claim:

The paper explains that by using self-attention instead of older methods like recurrent neural networks , a  model can process words all at once rather than one by one. This makes training much faster and fixes the common problem where models struggle to "remember" the beginning of a long sentence.

Ex - If we input a sentence “Awwab was kidnapped by a rogue gent  and taken to a basement during ARG “ in a model using RNN then the model fails to detect relation between various words like Awwab, kidnapper etc.  A transformer solves this problem and increases accuracy , RNN fails to keep track of previous words when too many words are given to a model.

---

## Core Architecture:

Embedding - Converts the input word into a vector  

Positional Encoder  - Creates positional information of each vector so that model knows,sequence of inputs  

Multi-Head Self-Attention (MHA) – Lets each word focus on other relevant words in the sentence from multiple perspectives.  

Feed-Forward Network (FFN) – Applies a small neural network to each token to refine and transform features.  

Encoder – Reads the input sentence and builds a  contextual representation of it.  

Decoder – Generates output step-by-step using encoded input + previously generated tokens.  

---

## Baseline and Dataset:

My code uses dialogue text files (input_texts.txt and label_texts.txt) imported from Kaggle.. Performance is evaluated using token-level classification accuracy and compared against standard sequential RNN baselines.
