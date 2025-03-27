import csv
import json
import os
from transformers import LongformerTokenizer, LongformerModel
import torch
from tqdm import tqdm

def embed_victorian_csv(
    csv_path,
    json_path="embeddings5.json",
    model_name="allenai/longformer-base-4096",
    cache_every=256,
    max_length=4096,
    max_rows=None  # ✅ New parameter
):
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    start_idx = len(existing_data) + 18950
    print(f"🔁 Resuming from row {start_idx}")
    with open(csv_path, mode="r", encoding="latin1") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if max_rows is not None:
        rows = rows[:max_rows]

    rows = rows[start_idx:]
    
    output = existing_data.copy()
    for i, row in enumerate(tqdm(rows, total=len(rows), desc="Embedding rows")):
        row = rows[i]
        text = row['text']
        author = row['author']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        output.append({
            "row": i + start_idx,
            "embedding": embedding,
            "author": author
        })
        if (i + 1) % cache_every == 0 or i == len(rows) - 1:
            with open(json_path, "w") as f:
                json.dump(output, f, indent=2)
    print(f"✅ Done! Total rows embedded: {len(output)}")

# Example usage
if __name__ == "__main__":
    embed_victorian_csv(
        "../Data/Gungor_2018_VictorianAuthorAttribution_data-train.csv",
        max_rows=None  # ✅ For quick testing; change to None on your PC
    )
