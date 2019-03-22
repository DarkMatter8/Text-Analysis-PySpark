from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.feature_extraction.text import TfidfVectorizer
import os 
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from pyspark.ml.clustering import KMeans
import findspark
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler

findspark.init("/usr/local/spark")

from pyspark.sql import SparkSession

spark = SparkSession.builder \
   .master("local") \
   .appName("Text analysis") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

titles = open('data/title_list.txt').read().split('\n')
synopses_wiki = open('data/synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]
genres = open('data/genres_list.txt').read().split('\n')
genres = genres[:100]

ranks = []
for i in range(1, len(titles)+1):
    ranks.append(i)

synopses_clean_wiki = []
for text in synopses_wiki:
    text = BeautifulSoup(text, 'html.parser').getText()
    synopses_clean_wiki.append(text)
synopses_wiki = synopses_clean_wiki

synopses_imdb = open('data/synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

synopses_clean_imdb = []

for text in synopses_imdb:
    text = BeautifulSoup(text, 'html.parser').getText()
    synopses_clean_imdb.append(text)
synopses_imdb = synopses_clean_imdb

synopses = []
for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

tfidf_matrix = pd.DataFrame(tfidf_matrix.todense())

print(type(tfidf_matrix))

sdf = sqlContext.createDataFrame(tfidf_matrix)

vecAssembler = VectorAssembler(inputCols=sdf.columns, outputCol="features")
new_df = vecAssembler.transform(sdf)

num_clusters = 5

kmeans = KMeans().setK(num_clusters).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(new_df)

transformed = model.transform(new_df).select('prediction')
rows = transformed.collect()

df_pred = sqlContext.createDataFrame(rows)

clusters = list(
    df_pred.select('prediction').toPandas()['prediction']
)

films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres }

frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])


frame['cluster'].value_counts()

grouped = frame['rank'].groupby(frame['cluster'])


similarity_distance = 1 - cosine_similarity(tfidf_matrix)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(similarity_distance)

xs, ys = pos[:, 0], pos[:, 1]

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

cluster_names = {0: 'Cluster 1', 
                 1: 'Cluster 2', 
                 2: 'Cluster 3', 
                 3: 'Cluster 4', 
                 4: 'Cluster 5'}

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 


groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9))

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=20, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=7)  

plt.show() #show the plot
