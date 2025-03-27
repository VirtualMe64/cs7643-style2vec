import gensim
from gensim.models import doc2vec
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io as sc
assert gensim.models.doc2vec.FAST_VERSION
import csv
from collections import namedtuple


train_data = pd.read_csv('Gungor_2018_VictorianAuthorAttribution_data-train.csv')
labeled_tuple = namedtuple('tuple', 'words tags')
train_texts = []
for i, text in tqdm(enumerate(train_data.text.values)):
    words = text.lower().split()
    index = [i]
    data = labeled_tuple(words,index)
    train_texts.append(data)
print(index)
print("embedding data...")
doc_embds = doc2vec.Doc2Vec(documents=train_texts, vector_size=762, window=50)

print("exporting embeddings...")
with open('embds_data.csv', 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['text embedding', 'index'])
    j=0
    for row in doc_embds:
            csv_out.writerow(row)
            j+=1
            if j == 18929:
                 break

#accessing an embedding from new csv
embds = pd.read_csv('doc2vec_embds.csv')
print(embds.values[0], len(embds.values[0]))      


