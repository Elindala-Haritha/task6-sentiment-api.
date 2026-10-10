
from pathlib import Path
import re

import joblib
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Keep training results reproducible
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Load the built-in movie review dataset
(train_x, train_y), (test_x, test_y) = imdb.load_data(
    num_words=10000
)

# Convert IMDb word-number sequences back into text
word_index = imdb.get_word_index()
index_word = {index + 3: word for word, index in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"


def decode_review(sequence):
    return " ".join(
        index_word.get(number, "<UNK>")
        for number in sequence
        if number >= 4
    )


train_texts = [decode_review(row) for row in train_x]
test_texts = [decode_review(row) for row in test_x]

# Convert review text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()

# Build a binary classification model
model = Sequential([
    Input(shape=(5000,)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    X_train,
    train_y,
    epochs=3,
    batch_size=128,
    validation_split=0.1
)

# Evaluate on reviews kept aside for testing
loss, accuracy = model.evaluate(X_test, test_y, verbose=1)
print(f"Test accuracy: {accuracy:.2%}")

# Save the model and vectorizer used by app.py
model.save(MODEL_DIR / "sentiment_model.keras")
joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

print("Training complete!")
print("Model files saved inside the model folder.")
