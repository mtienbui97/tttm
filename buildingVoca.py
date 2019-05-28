#building vocabulary bag
from collections import Counter
import time
import sys
import pandas as pd
import nltk
nltk.download('all')
nltk.download('punkt')
# choose PorterStemmer as stemmer
from nltk.stem import PorterStemmer
from nltk.corpus import words #import English corpus for comparing with data

mydataset = pd.read_csv('stemmed_data.csv',low_memory=False)

#tao 1 container de theo doi so lan gia tri tương duong dc them vao
all_words = Counter()

start = time.time()
progress = 0

#Count the number of occurrences of a word
def count_all(x):
    global start
    global all_words
    global  progress
    x = x.split(' ')
    for word in x:
        all_words[word] += 1
    progress += 1
    end = time.time()
    sys.stdout.write('\r {} percent, {} position, {} per second'.format((str(float(progress / len(mydataset)))),(progress), (1 / (end - start))))


start = time.time()
for item in mydataset.stems:
    count_all(item)

#create new data frame
allwords_df = pd.DataFrame(columns=['words', 'count'])
allwords_df['count'] = pd.Series(list(all_words.values()))
allwords_df['words'] = pd.Series(list(all_words.keys()))
allwords_df.index = allwords_df['words']
print(allwords_df.index)

#export to csv file
allwords_df.to_csv('all_words.csv')

#compare English corpus to our own
porter = PorterStemmer()
nltkstem = [porter.stem(word) for word in words.words()] #stem the #words in the NLTK corpus
# so that they’re equivalent to the words in #the allwordsdf dataframe

nltkwords = pd.DataFrame() #make nltkwords data frame

nltkwords['words'] = nltkstem

allwords_df = allwords_df[allwords_df['words'].isin(nltkwords['words'])] #keep only #those in the stemmed NLTK corpus
#export to csv file
allwords_df.to_csv('all_words_final.csv')