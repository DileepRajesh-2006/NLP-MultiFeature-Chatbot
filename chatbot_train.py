import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk

from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

with open("intents.json", "r") as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_letters
]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [
        lemmatizer.lemmatize(word.lower())
        for word in word_patterns
    ]

    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

training = np.array(training, dtype=object)

trainX = np.array(list(training[:, 0]))
trainY = np.array(list(training[:, 1]))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(trainX[0]),)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation="softmax")
])

sgd = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)

history = model.fit(
    trainX,
    trainY,
    epochs=250,
    batch_size=5,
    verbose=1
)

model.save("chatbot_model.keras")

print("Model training completed successfully!")