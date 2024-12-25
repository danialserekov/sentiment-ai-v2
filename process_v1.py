import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import time

# Load some the necessary resources for nltk
download('punkt') # for tokens to divide text for separate token
download('stopwords') # most recent words like ("and", "in", "on")

# Uploading csv dataset file with 500 thousand hundred line of reviews
data = pd.read_csv('Reviews.csv') # Amazon reviews Dataset 568,454 users

# Limit it for better capacity and fast loading
data = data.head(10000)  # first 10 thousand lines

# Dividing Sentiment for Positive / Neutral / Negative
def label_sentiment(score):
    if score >= 4:
        return 'Positive' # Score must be more or equal from 4
    elif score == 3:
        return 'Neutral' # Score must be equal to 3
    else:
        return 'Negative' # Score must be less than 3

data['Sentiment'] = data['Score'].apply(label_sentiment) # converts sentiments

# Tokenization process of text
def preprocess_text(text):
    tokens = word_tokenize(text.lower()) # converts text to lower case
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])

# Apply tokenization for text
data['ProcessedText'] = data['Text'].apply(preprocess_text)

# Split the data
X = data['ProcessedText']
y = data['Sentiment']

# Convert text to numeric signs
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Encode labels for all models
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Start timing for model training
start_time = time.time()

# Train Support Vector Machine
svm_model = SVC(kernel='linear', class_weight='balanced')

# Train Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

# Create a Hybrid Model using VotingClassifier (SVM and NN)
hybrid_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('nn', nn_model)
], voting='hard')

# Train the Hybrid Model
hybrid_model.fit(X_train, y_train)
hybrid_time = time.time() - start_time

# Evaluate the Hybrid Model
y_pred_hybrid = hybrid_model.predict(X_test)

print("Hybrid Model (SVM + NN) Classification Report:\n", classification_report(y_test, y_pred_hybrid, zero_division=0))

# Calculate execution time
print(f"\nExecution Time (seconds):")
print(f"Hybrid Model: {hybrid_time:.2f}s")

# Predict sample reviews using Hybrid Model
sample_reviews = [
    "This product is great! I love it.",
    "It was okay, not amazing, but decent.",
    "Worst purchase ever. Totally disappointed.",
    "Really good, but not perfect. I would recommend it.",
    "Absolutely terrible, I will never buy it again.",
    "Wtf is it",
    "Wow, beast",
    "Normal, clean",
    "Damn, that's cool",
    "Shit",
    "Yo, this thing slaps! Totally worth it.",
    "Bruh, waste of money. I'm mad disappointed.",
    "Aight, it's chill, but I seen better.",
    "Man, this is fire! 10/10 would cop again.",
    "Straight trash, don't bother. Smh.",
    "Sheesh, this product is clean, love it!",
    "Meh, it's whatever, not that deep.",
    "Fr? This is insane, best thing I've bought!",
    "Nah fam, not feeling it. Broke after a week.",
    "Lowkey thought it'd suck, but it's actually solid.",
    "This ain't it chief, I'm done with it.",
    "Hella good, legit the GOAT product."
]

# Sentiment Prediction
sample_reviews_processed = [preprocess_text(review) for review in sample_reviews]
sample_reviews_tfidf = vectorizer.transform(sample_reviews_processed)

sentiments_hybrid = hybrid_model.predict(sample_reviews_tfidf)

# Results
print("\nPredicted Sentiments for Sample Reviews (Hybrid Model):")
print(f"Hybrid Model (SVM + NN): {sentiments_hybrid}")
