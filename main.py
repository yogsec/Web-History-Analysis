import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load labeled data
df = pd.read_csv('labeled_data.csv')


# Improved URL preprocessing
def clean_url(url):
    url = re.sub(r'https?://(www\.)?', '', url)  # Remove protocol
    url = re.sub(r'\d+', '<NUM>', url)  # Replace numbers
    url = re.sub(r'[^a-zA-Z0-9./]', ' ', url)  # Keep slashes & dots
    return url.lower()


df['cleaned_url'] = df['url'].apply(clean_url)

# Convert categories to numbers
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_url'])
sequences = tokenizer.texts_to_sequences(df['cleaned_url'])
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['category_encoded'], test_size=0.2,
                                                    random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(len(df['category'].unique()), activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')


# Function to classify URLs from a file
def classify_urls_from_file(file_path):
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f.readlines()]

    cleaned_urls = [clean_url(url) for url in urls]
    sequences = tokenizer.texts_to_sequences(cleaned_urls)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    predictions = model.predict(padded)
    categories = [label_encoder.inverse_transform([np.argmax(pred)])[0] for pred in predictions]

    for url, category in zip(urls, categories):
        print(f'URL: {url} → Category: {category}')


# Classify URLs from an input file
url_file = input("Enter the filename containing URLs: ")
classify_urls_from_file(url_file)
