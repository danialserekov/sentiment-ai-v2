import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import time

# Load necessary resources for nltk
download('punkt')
download('stopwords')

# Load dataset
data = pd.read_csv('Reviews.csv')
data = data.head(10000)  # Limit dataset for efficiency


# Sentiment labeling function
def label_sentiment(score):
    if score >= 4:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Negative'


data['Sentiment'] = data['Score'].apply(label_sentiment)

# User profile features: Review count and average score per user
user_profiles = data.groupby('UserId').agg(ReviewCount=('Id', 'count'),
                                           AvgScore=('Score', 'mean')).reset_index()
data = pd.merge(data, user_profiles, on='UserId')


# Tokenization and preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])


data['ProcessedText'] = data['Text'].apply(preprocess_text)

# Split data
X = data['ProcessedText']
y = data['Sentiment']
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# Hybrid model setup
svm_model = SVC(kernel='linear', class_weight='balanced')
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
hybrid_model = VotingClassifier(estimators=[('svm', svm_model), ('nn', nn_model)], voting='hard')

# Train the hybrid model
start_time = time.time()
hybrid_model.fit(X_train, y_train)
hybrid_time = time.time() - start_time

# Evaluate overall model
y_pred_hybrid = hybrid_model.predict(X_test)
print("\nHybrid Model (Overall) Classification Report:")
print(classification_report(y_test, y_pred_hybrid, zero_division=0))
print(f"Hybrid Model Training Time: {hybrid_time:.2f} seconds")

# Evaluate by user profiles
user_segments = {
    'Low Activity': data[data['ReviewCount'] < 10],
    'Moderate Activity': data[(data['ReviewCount'] >= 10) & (data['ReviewCount'] < 50)],
    'High Activity': data[data['ReviewCount'] >= 50]
}

for segment_name, segment_data in user_segments.items():
    if segment_data.shape[0] > 0:  # Ensure the segment is not empty
        X_segment = vectorizer.transform(segment_data['ProcessedText'])
        y_segment = label_encoder.transform(segment_data['Sentiment'])
        y_pred_segment = hybrid_model.predict(X_segment)

        print(f"\nPerformance for {segment_name} Users:")
        print(classification_report(y_segment, y_pred_segment, zero_division=0))
        print(f"Accuracy: {accuracy_score(y_segment, y_pred_segment):.2f}")
    else:
        print(f"\nNo data available for {segment_name} Users. Skipping this segment.")

# Sample review predictions
sample_reviews = [
    "This product is amazing!",
    "Not great, not terrible.",
    "Horrible, never buying this again."
]
sample_reviews_processed = [preprocess_text(review) for review in sample_reviews]
sample_reviews_tfidf = vectorizer.transform(sample_reviews_processed)
sample_predictions = label_encoder.inverse_transform(hybrid_model.predict(sample_reviews_tfidf))
print("\nSample Review Predictions:", sample_predictions)