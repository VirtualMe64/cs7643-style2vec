import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import argparse
import os
from sklearn.model_selection import train_test_split

# --------- Fixed Hyperparameters ---------
hyperparams = {
    'margin': 0.5,
    'batch_size': 64,
    'epochs': 15,
    'model_path': 'Models/best_style_encoder.pt',
    'info_path': 'Models/best_model_info.json',
    'log_path': 'Models/all_model_results.json',
    'train_split': 0.7,
    'cv_split': 0.15,
    'test_split': 0.15,
}

# --------- Dataset + Encoder ---------
class TripletDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.author_to_indices = df.groupby('author').indices
        self.embeddings = np.stack(df['embedding'].values)
        self.authors = df['author'].values

    def __getitem__(self, idx):
        anchor = self.embeddings[idx]
        anchor_author = self.authors[idx]

        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.author_to_indices[anchor_author])
        positive = self.embeddings[pos_idx]

        neg_author = anchor_author
        while neg_author == anchor_author:
            neg_author = random.choice(list(self.author_to_indices.keys()))
        neg_idx = random.choice(self.author_to_indices[neg_author])
        negative = self.embeddings[neg_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
            anchor_author
        )

    def __len__(self):
        return len(self.df)

def build_encoder(input_dim, style_dim, num_layers, dropout=0.3):
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(input_dim, style_dim))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        input_dim = style_dim
    return nn.Sequential(*layers)

class StyleEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.net = encoder_net

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

# --------- Load Data ---------
def load_master_embeddings(path='Data/masterembeddings.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['embedding'] = df['embedding'].apply(np.array)
    return df

def split_dataset(df):
    authors = df['author'].unique()
    train_auth, test_auth = train_test_split(authors, test_size=(hyperparams['cv_split'] + hyperparams['test_split']), random_state=42)
    cv_auth, test_auth = train_test_split(test_auth, test_size=hyperparams['test_split'] / (hyperparams['test_split'] + hyperparams['cv_split']), random_state=42)

    train_df = df[df['author'].isin(train_auth)]
    cv_df = df[df['author'].isin(cv_auth)]
    test_df = df[df['author'].isin(test_auth)]
    return train_df, cv_df, test_df

# --------- Evaluation ---------
def evaluate(encoder, df):
    encoder.eval()
    dataset = TripletDataset(df)
    loader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    vectors = []
    labels = []
    with torch.no_grad():
        for anchor, _, _, author in loader:
            style = encoder(anchor)
            vectors.extend(style.cpu().numpy())
            labels.extend(author)

    from collections import defaultdict
    from sklearn.metrics.pairwise import cosine_similarity

    author_to_vecs = defaultdict(list)
    for vec, label in zip(vectors, labels):
        author_to_vecs[label].append(vec)

    centroids = {k: np.mean(vs, axis=0) for k, vs in author_to_vecs.items()}

    correct = 0
    for vec, label in zip(vectors, labels):
        sims = {k: cosine_similarity([vec], [centroid])[0][0] for k, centroid in centroids.items()}
        pred = max(sims, key=sims.get)
        if pred == label:
            correct += 1

    return correct / len(vectors)

# --------- Training ---------
import time

def train():
    df = load_master_embeddings()
    train_df, cv_df, test_df = split_dataset(df)
    input_dim = len(train_df['embedding'].iloc[0])

    style_dims = [128, 256, 512, 762, 1024]
    layer_counts = [1, 2, 3]
    lrs = [1e-5, 5e-5, 1e-4]
    results = []
    best_cv_acc = -1

    start_time = time.time()

    for style_dim in style_dims:
        for layers in layer_counts:
            for lr in lrs:
                encoder_net = build_encoder(input_dim, style_dim, layers, dropout=0.3)
                encoder = StyleEncoder(encoder_net)
                optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
                loss_fn = nn.TripletMarginLoss(margin=hyperparams['margin'], p=2)
                train_loader = DataLoader(TripletDataset(train_df), batch_size=hyperparams['batch_size'], shuffle=True)

                for epoch in range(hyperparams['epochs']):
                    encoder.train()
                    for anchor, pos, neg, _ in train_loader:
                        optimizer.zero_grad()
                        a_out = encoder(anchor)
                        p_out = encoder(pos)
                        n_out = encoder(neg)
                        loss = loss_fn(a_out, p_out, n_out)
                        loss.backward()
                        optimizer.step()

                start = time.time()
                cv_acc = evaluate(encoder, cv_df)
                train_time = round(time.time() - start)
                result = {
                    'style_dim': style_dim,
                    'layers': layers,
                    'learning_rate': lr,
                    'cv_accuracy': cv_acc,
                    'train_time_seconds': train_time
                }
                results.append(result)
                print(result)

                if cv_acc > best_cv_acc:
                    best_cv_acc = cv_acc
                    torch.save(encoder.state_dict(), hyperparams['model_path'])
                    with open(hyperparams['info_path'], 'w') as f:
                        json.dump(result, f, indent=2)

    with open(hyperparams['log_path'], 'w') as f:
        json.dump(results, f, indent=2)

# --------- Testing ---------
def test():
    df = load_master_embeddings()
    _, _, test_df = split_dataset(df)
    input_dim = len(test_df['embedding'].iloc[0])

    with open(hyperparams['info_path'], 'r') as f:
        best_params = json.load(f)

    encoder_net = build_encoder(input_dim, best_params['style_dim'], best_params['layers'], dropout=0.3)
    encoder = StyleEncoder(encoder_net)
    encoder.load_state_dict(torch.load(hyperparams['model_path']))
    test_acc = evaluate(encoder, test_df)
    print(f"Test Accuracy: {test_acc:.4f}")

# --------- Main ---------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
