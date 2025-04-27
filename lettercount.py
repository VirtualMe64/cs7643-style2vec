import csv
from tqdm import tqdm
from collections import Counter

train_data = []
test_data = []

def is_test(idx):
    return idx % 10 == 0

with open("./Data/Gungor_2018_VictorianAuthorAttribution_data-train.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    next(reader)
    for i, row in enumerate(reader):
        if is_test(i):
            test_data.append(row)
        else:
            train_data.append(row)
    
class LetterFrequencyClassifier:
    def __init__(self):
        self.author_vecs = {}
        self.author_counts = {}

    def _vectorize(self, text):
        counts = Counter(text)
        letters = list('abcdefghijklmnopqrstuvwxyz')
        vec = [counts[letter] for letter in letters]
        total_letters = sum(vec)
        vec = [count / total_letters for count in vec]
        return vec
    
    def update(self, text, author):
        vec = self._vectorize(text)
        if author in self.author_vecs:
            self.author_vecs[author] = [a + b for a, b in zip(self.author_vecs[author], vec)]
            self.author_counts[author] += 1
        else:
            self.author_vecs[author] = vec
            self.author_counts[author] = 1
    
    def predict(self, text):
        vec = self._vectorize(text)
        target_vecs = {author: [a / self.author_counts[author] for a in self.author_vecs[author]] for author in self.author_vecs}
        scores = {author: sum([abs(a - b) ** 2 for a, b in zip(vec, target_vecs[author])]) for author in target_vecs}
        return min(scores.keys(), key = lambda x: scores[x])

classifier = LetterFrequencyClassifier()
for row in tqdm(train_data, desc = "Training"):
    classifier.update(row[0], row[1])

correct = 0
total = 0
for row in tqdm(test_data, desc="Testing"):
    total += 1
    if classifier.predict(row[0]) == row[1]:
        correct += 1
print(correct / total)