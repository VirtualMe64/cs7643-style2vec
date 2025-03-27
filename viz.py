import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
from train_neural import build_encoder, StyleEncoder, hyperparams

def load_train_set(path='Data/masterembeddings.json', max_per_author=1000):
    with open(path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(data)
    df['embedding'] = df['embedding'].apply(np.array)
    authors = df['author'].unique()
    train_auth, test_cv_auth = train_test_split(authors, test_size=(hyperparams['cv_split'] + hyperparams['test_split']), random_state=42)

    train_df = df[df['author'].isin(train_auth)].reset_index(drop=True)

    # Limit to max_per_author rows per author
    balanced = train_df.groupby('author').apply(lambda x: x.sample(n=min(len(x), max_per_author), random_state=42)).reset_index(drop=True)
    return balanced

def run_tsne(vectors, labels, title='t-SNE', filename=None):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(np.stack(vectors))

    label_to_color = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    colors = [label_to_color[label] for label in labels]

    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='tab20', s=10)
    plt.title(title)
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Saved plot to {filename}")
    else:
        plt.show()

def main():
    df = load_train_set()
    vectors_raw = df['embedding'].tolist()
    labels = df['author'].tolist()

    print('Running t-SNE on raw embeddings...')
    run_tsne(vectors_raw, labels, title='t-SNE of Raw LLM Embeddings', filename='tsne_raw.png')

    # Load best model
    with open(hyperparams['info_path'], 'r') as f:
        best_params = json.load(f)
    input_dim = len(vectors_raw[0])
    encoder_net = build_encoder(input_dim, best_params['style_dim'], best_params['layers'], dropout=0.3)
    encoder = StyleEncoder(encoder_net)
    encoder.load_state_dict(torch.load(hyperparams['model_path']))
    encoder.eval()

    print('Generating style vectors...')
    style_vectors = []
    with torch.no_grad():
        for vec in vectors_raw:
            vec_tensor = torch.tensor([vec], dtype=torch.float32)
            style = encoder(vec_tensor).cpu().numpy()[0]
            style_vectors.append(style)

    print('Running t-SNE on style vectors...')
    run_tsne(style_vectors, labels, title='t-SNE of Style Vectors (Post-Encoder)', filename='tsne_styled.png')

if __name__ == '__main__':
    main()
