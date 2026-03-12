import nltk
from nltk.tokenize import word_tokenize
import string
import os
import random
import pickle
import math
from typing import Optional

corpus = []
for p, d, fs in os.walk("Train_Full"):
    for f in fs:
        if not f.endswith("txt"):
            continue
        fp = os.path.join(p, f)
        with open(fp, "r", encoding='utf-16') as fi:
            t = fi.read().split("\n")
            t = [s for s in t if len(s.strip()) > 0]
            corpus.extend(t)

clean_corpus = []
bigrams = []

for sentence in corpus:
    tokens = word_tokenize(sentence)
    tokens = ["<s>"] + tokens + ["</s>"]
    bigrams.extend(nltk.bigrams(tokens))

word_count = {}

for s, e in bigrams:
    if s not in word_count:
        word_count[s] = {}
    if e not in word_count[s]:
        word_count[s][e] = 0
    
    word_count[s][e] += 1

# model = {}
# for s in word_count:
#     most_common = ""
#     most_common_count = 0
#     for e, c in word_count[s].items():
#         if c > most_common_count:
#             most_common_count = c
#             most_common = e
#     model[s] = e

# cur = "<s>"
# print(cur, end=" ")
# while cur != "</s>":
#     cur = model[cur]
#     print(cur, end=" ")
# print("")



model = {}
for w, wc in word_count.items():
    tot = 0
    for n, nc in wc.items():
        tot += nc
    model[w] = {n: math.log(nc / tot) for n, nc in wc.items()}

# Save the model to a file using pickle
# with open("corpus_model", "wb") as f:
#     pickle.dump(model, f)

log_dict = model

# with open("corpus_model", "rb") as f:
#     log_dict = pickle.load(f)

# print(log_dict["<s>"]["Hôm"])
# print(log_dict["Hôm"]["nay"])
# print(log_dict["nay"]["trời"])
# print(log_dict["trời"]["đẹp"])
# print(log_dict["đẹp"]["lắm"])
# print(log_dict["lắm"]["."])
# print(log_dict["."]["</s>"])

# Probability of the sentence "Hôm nay trời đẹp lắm ."
sentence = ["<s>", "Hôm", "nay", "trời", "đẹp", "lắm", ".", "</s>"]
sentence_log_prob = 0.0
for i in range(len(sentence) - 1):
    w1, w2 = sentence[i], sentence[i + 1]
    if w1 in log_dict and w2 in log_dict[w1]:
        sentence_log_prob += log_dict[w1][w2]
    else:
        sentence_log_prob += math.log(1e-10)  # Smoothing for unseen bigrams
print(f"Log probability of the sentence: {sentence_log_prob}")
# Print probability of the sentence
print(f"Probability of the sentence: {math.exp(sentence_log_prob)}")

class Model:

    def __init__(self, log_dict: dict[str, dict[str, float]]):
        self.model: dict[str, list[tuple[float, str]]] = {}
        self.log_dict = log_dict

        for w, wns in log_dict.items():
            self.model[w] = []
            for wn, wnp in wns.items():
                self.model[w].append((wnp, wn))
            self.model[w].sort(reverse=True)
    
    def generate_next_word(self, word: str, n: int=10) -> str:
        '''
        Generate the next word based on the log probabilities of next top n words
        '''
        if word not in self.model:
            return "</s>"
        candidates = self.model[word][:n]

        # Calculate the probabilities from log probabilities
        log_sum = math.log(sum(math.exp(log_p) for log_p, _ in candidates))
        probabilities = [(math.exp(log_p - log_sum), w) for log_p, w in candidates]

        # Randomly select the next word based on the probabilities (used random.choices for weighted random selection)
        return random.choices([w for _, w in probabilities], weights=[p for p, _ in probabilities])[0]

model = Model(log_dict)

def generate():
    cur = "<s>"
    print(cur, end=" ")
    while cur != "</s>":
        cur = model.generate_next_word(cur, n=5)
        print(cur, end=" ")
    print("")
for i in range(20):
    generate()