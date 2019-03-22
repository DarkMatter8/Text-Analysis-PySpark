# Text-Analysis-PySpark
Clustering of top 100 IMDB movies using Text Analysis in Apache Spark.

### Technologies Used
1. Apache Spark
2. PySpark
3. nltk
4. sklearn

### Installing Apache Spark
[Apache Spark Installation Medium](https://medium.com/devilsadvocatediwakar/installing-apache-spark-on-ubuntu-8796bfdd0861)

### Installing PySpark 
```
pip install pyspark
```
### Code

#### Initializing Spark Session
```
findspark.init("/usr/local/spark")

from pyspark.sql import SparkSession

spark = SparkSession.builder \
   .master("local") \
   .appName("Text analysis") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)
```  

#### Importing data
```
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

```

#### Data Preprocessing
```
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

```

#### Generating Tf-Idf Matrix
```
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

tfidf_matrix = pd.DataFrame(tfidf_matrix.todense())
```

#### Converting to PySpark Dataframe
```
sdf = sqlContext.createDataFrame(tfidf_matrix)

vecAssembler = VectorAssembler(inputCols=sdf.columns, outputCol="features")
new_df = vecAssembler.transform(sdf)
``` 

#### Applying KMeans to form clusters
```
num_clusters = 5

kmeans = KMeans().setK(num_clusters).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(new_df)

transformed = model.transform(new_df).select('prediction')
rows = transformed.collect()

df_pred = sqlContext.createDataFrame(rows)

clusters = list(
    df_pred.select('prediction').toPandas()['prediction']
)

```

### Plotting the result
![result](https://github.com/DarkMatter8/Text-Analysis-PySpark/blob/master/result.png)