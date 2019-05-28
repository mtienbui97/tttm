# import data from file
import pandas as pd
import numpy as np

#mydataset = pd.read_csv('articles.csv', low_memory=False)
mydataset = pd.DataFrame(columns=['id','title','author','date','content','year','month','publication','category','digital','section','url'])
chunksize = 10**6
for chunk in pd.read_csv('articles.csv', chunksize=chunksize ):
    mydataset = mydataset.append(chunk)
print(mydataset)

# import the nltk package
import nltk

nltk.download('all')
nltk.download('punkt')
# call the nltk downloader
# nltk.download()
# import pos_tag for specific tag for certain words
from nltk import pos_tag
# choose PorterStemmer as stemmer
from nltk.stem import PorterStemmer
# import tokenize to split content into token
from nltk.tokenize import sent_tokenize, word_tokenize
# not read yet
import time
import re
import sys

porter = PorterStemmer()
# choose different corpus and datasets
# tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

progress = 0  # for keeping track of where the function is


def stem(x):
    end = time.time()

    # split words
    token_words = word_tokenize(x)
    # array of tokens
    tokens = []
    for word in token_words:
        # skip '...'
        if word.strip('.') == '':
            pass
        elif re.search(r'\d+', word):  # getting rid of digits
            pass
        else:
            tokens.append(word.strip('.'))
    #print(tokens)
    global start
    global progress
    # specify token with pos_tag
    tokens = pos_tag(tokens)
    progress += 1
    # reduce each words to  root form
    stem_content = []
    for tagged_word in tokens:
        word = porter.stem(tagged_word[0])
        word.lower()
        if word != 'NNP':
            stem_content.append(porter.stem(word))
            stem_content.append(" ")
    return "".join(stem_content)

    end = time.time()

    # lets us see how much time is left

    start = time.time()
    return stem_content


# x = "And never... and fulfilled a prophecy in the process. In the second season finale, back in 1991"
# print(stem(x))
start = time.time()
print(mydataset.content.apply(lambda x: stem(x)))
mydataset['stems'] = mydataset.content.apply(lambda x: stem(x))

mydataset.to_csv('stemmed_articles.csv')
