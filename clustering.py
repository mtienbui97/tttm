from sklearn.cluster import KMeans
import pandas as pd


vec_matrix_pca = pd.read_csv('vec_matrix_pca.csv', low_memory=False)
word_matrix_pca = pd.read_csv('word_matrix.csv', low_memory=False)
mydataset = pd.read_csv('test_data.csv', low_memory=False)
words_df = pd.read_csv('all_words_final.csv', low_memory=False)

vec_clf10 = KMeans(n_clusters=10, verbose=0)
word_clf10 = KMeans(n_clusters=10,verbose=0)
vec_clf10.fit(vec_matrix_pca)
word_clf10.fit(word_matrix_pca)
#assign the label
mydataset['labels'] = vec_clf10.labels_
words_df['labels'] = word_clf10.labels_

labelsdf = mydataset.groupby(['publication', 'labels']).count()
publist = list(mydataset['publication'].unique())
print(publist)
words_df.to_csv('foo.csv')
