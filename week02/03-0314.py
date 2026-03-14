# 1) Hãy tải tập dữ liệu Brown corpus (từ NLTK)
# 2) Hãy thực hiện gán nhãn từ loại trên Brown corpus bằng 2 bộ POS tagger
# 3) Hãy đánh giá độ chính xác của 2 bộ POS tagger trên tập Brown corpus sử dụng các độ đo
# (precision, recall và macro-F1)

# USING ONLY ENGLISH BELOW THIS POINT PLEASE (TRY NOT TO HAVE OVER 100 characters each rows)

# POS tagger

# Download Brown corpus from nltk
import nltk
from nltk.corpus import brown
from nltk import pos_tag_sents, pos_tag
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from nltk.tag import brill
from nltk.tag.brill_trainer import BrillTaggerTrainer

import spacy
from spacy.tokens import Doc

import matplotlib.pyplot as plt

nltk.download('brown')
nltk.download('punkt')
nltk.download('universal_tagset')

# Load Brown corpus
sentences = brown.sents()

true_tags = []
for sentence in brown.tagged_sents(tagset="universal"):
    true_tags.extend(tag for _, tag in sentence)
labels = sorted(set(true_tags))

# Function for reporting
def report_evaluation(true, predicted):
    # Calculate accuracy
    accuracy = accuracy_score(true, predicted)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(true, predicted, average='macro', zero_division=0)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# NLTK default tag
predicted_tags = []
for sentence in pos_tag_sents(sentences, tagset="universal"):
    predicted_tags.extend(tag for _, tag in sentence)

print("NLTK pos tagger evaluation: ")
report_evaluation(true_tags, predicted_tags)

# Spacy
nlp = spacy.load('en_core_web_sm', disable=["parser","ner","lemmatizer"])
predicted_tags_spacy = []
docs = (Doc(nlp.vocab, words=sent) for sent in sentences)

i = 0
n = 5000
l = len(sentences)

# Function to normalized spacy tag (as tag. may be a little different)
SPACY_TO_BROWN = {
    "PART": "PRT",
    "PUNCT": ".",
    "AUX": "VERB",
    "CCONJ": "CONJ",
    "SCONJ": "CONJ",
    "PROPN": "NOUN",
    "INTJ": "X",
    "SYM": "X"
}

def normalize_spacy_tag(tag):
    return SPACY_TO_BROWN.get(tag, tag)

# Start predicting with spacy
print("Start tagging with spacy:")
for doc in nlp.pipe(docs, batch_size=n):
    predicted_tags_spacy.extend(normalize_spacy_tag(t.pos_) for t in doc)
    i += 1
    if i % (4 * n) == 0 or i == l:
        print(f"Finished tagging {i}/{l} sentences")

print("Spacy tagger evaluation: ")


report_evaluation(true_tags, predicted_tags_spacy)

labels_spacy = sorted(set(predicted_tags_spacy))
# Using matplotlib to draw heat map of both tagger prediction compare to true tags and save it

cm_nltk = confusion_matrix(true_tags, predicted_tags, labels=labels)
cm_spacy = confusion_matrix(true_tags, predicted_tags_spacy, labels=labels_spacy)

# Normalized each true label to 1
cm_nltk = cm_nltk / cm_nltk.sum(axis=1, keepdims=True)
cm_spacy = cm_spacy / cm_spacy.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

def draw_heatmap(ax, cm, labels, title):
    im = ax.imshow(cm, cmap="gray")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # threshold for luminance
    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]

            color = "white" if value < threshold else "black"

            ax.text(
                j, i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8
            )

# draw both heatmaps
draw_heatmap(axes[0], cm_nltk, labels, "NLTK POS Tagger")
draw_heatmap(axes[1], cm_spacy, labels_spacy, "spaCy POS Tagger")

plt.tight_layout()
plt.savefig("03-0314_pos_tagger_comparison.png")
# plt.show()