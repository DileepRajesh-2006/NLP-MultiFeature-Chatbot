import re
import pickle
import nltk
import os
import urllib.request

from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import numpy as np

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("brown", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(words)

os.makedirs("data", exist_ok=True)

SST_BASE = (
    "https://raw.githubusercontent.com/"
    "AcademiaSinicaNLPLab/sentiment_dataset/master/data/"
)

SST_FILES = {
    "data/sst_train.txt": "stsa.binary.train",
    "data/sst_dev.txt": "stsa.binary.dev",
    "data/sst_test.txt": "stsa.binary.test",
}

for local_path, remote_name in SST_FILES.items():
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(SST_BASE + remote_name, local_path)

def load_sst(filepath):
    texts = []
    labels = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            label = int(line[0])
            text = line[2:].strip()

            texts.append(clean_text(text))
            labels.append(label)

    return texts, labels

train_texts, train_labels = load_sst("data/sst_train.txt")
dev_texts, dev_labels = load_sst("data/sst_dev.txt")
test_texts, test_labels = load_sst("data/sst_test.txt")

train_texts = train_texts + dev_texts
train_labels = train_labels + dev_labels

brown_sents = brown.sents(categories=["news", "government", "hobbies"])

neutral_texts = []
neutral_labels = []

for sent in brown_sents:
    text = " ".join(sent)
    cleaned = clean_text(text)

    if len(cleaned.split()) >= 4:
        neutral_texts.append(cleaned)
        neutral_labels.append(2)

target_neutral = len(train_texts) // 2

neutral_texts = neutral_texts[:target_neutral]
neutral_labels = neutral_labels[:target_neutral]

all_train_texts = train_texts + neutral_texts[:int(target_neutral * 0.9)]
all_train_labels = train_labels + neutral_labels[:int(target_neutral * 0.9)]

all_test_texts = test_texts + neutral_texts[int(target_neutral * 0.9):]
all_test_labels = test_labels + neutral_labels[int(target_neutral * 0.9):]

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(all_train_texts)
X_test = vectorizer.transform(all_test_texts)

model = LogisticRegression(
    max_iter=1000,
    C=1.5,
    solver="lbfgs"
)

model.fit(X_train, all_train_labels)

predictions = model.predict(X_test)

accuracy = accuracy_score(all_test_labels, predictions)

print(accuracy)

print(
    classification_report(
        all_test_labels,
        predictions,
        target_names=["Negative", "Positive", "Neutral"]
    )
)

test_sentences = [
    "I love this workout, it feels amazing!",
    "My knee hurts so much, this is terrible.",
    "The gym opens at 9am on weekdays.",
    "I am so happy with my progress!",
    "I hate this pain, it never goes away.",
]

label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}

for s in test_sentences:
    vec = vectorizer.transform([clean_text(s)])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    conf = round(max(prob) * 100, 1)

    print(s)
    print(label_map[pred], conf)

pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))