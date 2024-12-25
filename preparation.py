import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import time

# Download necessary NLTK resources
download('punkt')
download('stopwords')

# Load dataset
data = pd.read_csv('Reviews.csv')

# Limit the dataset for faster execution
data = data.head(10000)

# Label sentiments
def label_sentiment(score):
    if score >= 4:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Negative'

data['Sentiment'] = data['Score'].apply(label_sentiment)

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])

data['ProcessedText'] = data['Text'].apply(preprocess_text)

# Split the data
X = data['ProcessedText']
y = data['Sentiment']

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Encode labels for all models
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Start timing for model training
start_time = time.time()

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=10000, class_weight='balanced')
lr_model.fit(X_train, y_train)
lr_time = time.time() - start_time

# Train Support Vector Machine
start_time = time.time()
svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train, y_train)
svm_time = time.time() - start_time

# Train Neural Network
start_time = time.time()
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
nn_model.fit(X_train, y_train)
nn_time = time.time() - start_time

# Train Random Forest
start_time = time.time()
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

# Train XGBoost
start_time = time.time()
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train_encoded)
xgb_time = time.time() - start_time

# Evaluate models
y_pred_lr = lr_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb_encoded = xgb_model.predict(X_test)
y_pred_xgb = label_encoder.inverse_transform(y_pred_xgb_encoded)

print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr, zero_division=0))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=0))
print("Neural Network Classification Report:\n", classification_report(y_test, y_pred_nn, zero_division=0))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb, zero_division=0))

# Calculate execution times
print(f"\nExecution Times (seconds):")
print(f"Logistic Regression: {lr_time:.2f}s")
print(f"SVM: {svm_time:.2f}s")
print(f"Neural Network: {nn_time:.2f}s")
print(f"Random Forest: {rf_time:.2f}s")
print(f"XGBoost: {xgb_time:.2f}s")

# Predict sample reviews
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

sample_reviews_processed = [preprocess_text(review) for review in sample_reviews]
sample_reviews_tfidf = vectorizer.transform(sample_reviews_processed)

sentiments_lr = lr_model.predict(sample_reviews_tfidf)
sentiments_svm = svm_model.predict(sample_reviews_tfidf)
sentiments_nn = nn_model.predict(sample_reviews_tfidf)
sentiments_rf = rf_model.predict(sample_reviews_tfidf)
sentiments_xgb_encoded = xgb_model.predict(sample_reviews_tfidf)
sentiments_xgb = label_encoder.inverse_transform(sentiments_xgb_encoded)

print("\nPredicted Sentiments:")
print(f"Logistic Regression: {sentiments_lr}")
print(f"SVM: {sentiments_svm}")
print(f"Neural Network: {sentiments_nn}")
print(f"Random Forest: {sentiments_rf}")
print(f"XGBoost: {sentiments_xgb}")
