from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

import pandas as pd
allwordsdf = pd.read_csv('all_words_final.csv', low_memory=False)
mydataset = pd.read_csv('stemmed_data.csv', low_memory=False)


#create stopwords
stopwords = list(allwordsdf[(allwordsdf['count'] >= allwordsdf['count'].quantile(.995))
            | (allwordsdf['count'] <= allwordsdf['count'].quantile(.4))]['words'])
print('stop: ')
print(stopwords)
#df=pd.DataFrame({'stopword': stopwords})


#create voca list
vecvocab = list(allwordsdf[(allwordsdf['count'] < allwordsdf['count'].quantile(.995))
                           & (allwordsdf['count'] > allwordsdf['count'].quantile(.4))]['words'])

print()
print('len vocab: ')
print(len(vecvocab))
#df=pd.DataFrame({'vocabword': vecvocab})

vec = TfidfVectorizer(stop_words=stopwords, vocabulary=vecvocab, tokenizer=None)
print(vec)

#transform the dataframe
vec_matrix = vec.fit_transform(mydataset['stems'])
word_matrix = vec.fit_transform(allwordsdf['words'])
print(vec_matrix)
print(type(vec_matrix))

#reduce dimension
pca = TruncatedSVD(n_components=100)
vec_matrix_pca = pca.fit_transform(vec_matrix)
word_matrix_pca = pca.fit_transform(word_matrix)
print(vec_matrix_pca)
print(len(vec_matrix_pca))


vec_matrix_pca_df = pd.DataFrame(vec_matrix_pca)
vec_matrix_pca_df.to_csv('vec_matrix_pca.csv')

word_matrix_df = pd.DataFrame(word_matrix_pca)
word_matrix_df.to_csv('word_matrix.csv')