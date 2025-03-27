import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def load_raw_embeddings(path='Data/masterembeddings.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['embedding'] = df['embedding'].apply(np.array)
    return df

def split_dataset(df, train_ratio=0.7, cv_ratio=0.15, test_ratio=0.15):
    authors = df['author'].unique()
    train_auth, test_auth = train_test_split(authors, test_size=(cv_ratio + test_ratio), random_state=42)
    cv_auth, test_auth = train_test_split(test_auth, test_size=test_ratio / (cv_ratio + test_ratio), random_state=42)

    train_df = df[df['author'].isin(train_auth)]
    cv_df = df[df['author'].isin(cv_auth)]
    test_df = df[df['author'].isin(test_auth)]
    return train_df, cv_df, test_df

def evaluate_baseline(test_df, ref_df):
    author_to_vecs = defaultdict(list)
    for _, row in ref_df.iterrows():
        author_to_vecs[row['author']].append(row['embedding'])

    # Compute centroids for each author
    centroids = {k: np.mean(vs, axis=0) for k, vs in author_to_vecs.items()}

    # Predict based on nearest centroid
    correct = 0
    for _, row in test_df.iterrows():
        vec = row['embedding']
        sims = {k: cosine_similarity([vec], [centroid])[0][0] for k, centroid in centroids.items()}
        pred = max(sims, key=sims.get)
        print(pred, row['author'])
        if pred == row['author']:
            correct += 1

    return correct / len(test_df)

if __name__ == "__main__":
    df = load_raw_embeddings()
    train_df, _, test_df = split_dataset(df)
    baseline_acc = evaluate_baseline(test_df, train_df)
    print(f"Baseline accuracy using raw embeddings: {baseline_acc:.4f}")
