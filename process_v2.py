import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
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
data = data.head(1000)


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

# Define individual models
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
svm_model = SVC(kernel='linear', class_weight='balanced', probability=True)
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the models
models = [lr_model, svm_model, nn_model, rf_model, xgb_model]
for model in models:
    model.fit(X_train, y_train_encoded)

# Predict with all models
predictions = np.zeros((X_test.shape[0], len(models)))  # Use shape[0] instead of len(X_test)

for i, model in enumerate(models):
    predictions[:, i] = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class

# Average the probabilities (soft voting)
final_predictions = (predictions.mean(axis=1) > 0.5).astype(int)  # Majority vote for final prediction

# Evaluate the ensemble model
print("Custom Ensemble Model (Soft Voting) Classification Report:\n",
      classification_report(y_test_encoded, final_predictions, zero_division=0, target_names=label_encoder.classes_))

# Predict sample reviews using the custom ensemble model
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

# Get final predictions for sample reviews
predictions_sample = np.zeros((sample_reviews_tfidf.shape[0], len(models)))  # Use shape[0] instead of len()

for i, model in enumerate(models):
    predictions_sample[:, i] = model.predict_proba(sample_reviews_tfidf)[:, 1]

final_predictions_sample = (predictions_sample.mean(axis=1) > 0.5).astype(int)
sentiments_sample_decoded = label_encoder.inverse_transform(final_predictions_sample)

print("\nPredicted Sentiments for Sample Reviews (Custom Ensemble):")
for review, sentiment in zip(sample_reviews, sentiments_sample_decoded):
    print(f"{review} -> {sentiment}")