#natural language toolkit

#import libraries

import numpy as np

import nltk
nltk.download('punkt')

from nltk import stem

stemmer= stem.PorterStemmer()

#tokenization
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#stemming
def stem(word):
    return stemmer.stem(word.lower())

#bagging
def bag_of_words(tokenized_sentence, words):
    sentence_words= [stem(word) for word in tokenized_sentence]
    
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx]= 1

    return bag
