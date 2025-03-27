import json
import os

def load_json_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def load_json_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all_embeddings(folder_path):
    all_data = []

    # Load 1, 3, 4, 5 (list-of-dict format)
    for i in [1, 3, 4, 5]:
        path = os.path.join(folder_path, f'embeddings{i}.json')
        data = load_json_list(path)
        all_data.extend(data)

    path2 = os.path.join(folder_path, 'embeddings2.json')
    data2 = load_json_lines(path2)
    all_data.extend(data2)
    len1 = len(load_json_list(os.path.join(folder_path, 'embeddings1.json')))
    len2 = len(load_json_lines(os.path.join(folder_path, 'embeddings2.json')))
    offset = len1 + len2

    embeddings3_path = os.path.join(folder_path, 'embeddings3.json')
    embeddings3_fixed = []
    for i, item in enumerate(load_json_list(embeddings3_path)):
        item['row'] = offset + i
        embeddings3_fixed.append(item)
    final_data = []
    final_data.extend(load_json_list(os.path.join(folder_path, 'embeddings1.json')))
    final_data.extend(data2)
    final_data.extend(embeddings3_fixed)
    final_data.extend(load_json_list(os.path.join(folder_path, 'embeddings4.json')))
    final_data.extend(load_json_list(os.path.join(folder_path, 'embeddings5.json')))

    return final_data

def save_master_embeddings(data, output_path='masterembeddings.json'):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"Saved master embeddings to {output_path} with {len(data)} entries.")

if __name__ == '__main__':
    folder = '.'  # Replace with your actual path if needed
    master_data = load_all_embeddings(folder)
    save_master_embeddings(master_data)
