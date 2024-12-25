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

# Load necessary resources for nltk
download('punkt')  # for tokens to divide text for separate token
download('stopwords')  # most recent words like ("and", "in", "on")

# Uploading the csv dataset
data = pd.read_csv('Reviews.csv')  # Amazon reviews Dataset 568,454 users

# Limit it for better capacity and faster loading
data = data.head(1000)  # first 10 thousand lines


# Dividing Sentiment for Positive / Neutral / Negative
def label_sentiment(score):
    if score >= 4:
        return 'Positive'  # Score must be more or equal from 4
    elif score == 3:
        return 'Neutral'  # Score must be equal to 3
    else:
        return 'Negative'  # Score must be less than 3


data['Sentiment'] = data['Score'].apply(label_sentiment)  # converts sentiments


# Tokenization process of text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # converts text to lower case
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])


# Apply tokenization for text
data['ProcessedText'] = data['Text'].apply(preprocess_text)

# Split the data
X = data['ProcessedText']
y = data['Sentiment']

# Convert text to numeric features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Encode labels for all models
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data based on user profiles
user_profiles = data['ProfileName'].unique()  # Get unique profiles

# Initialize the model
svm_model = SVC(kernel='linear', class_weight='balanced')
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
hybrid_model = VotingClassifier(estimators=[('svm', svm_model), ('nn', nn_model)], voting='hard')

# Initialize timing
start_time = time.time()

# Store results for each user profile
results = {}

# Train and evaluate the model for each user profile
for profile in user_profiles:
    print(f"\nProcessing user profile: {profile}")

    # Filter data for the current profile
    profile_data = data[data['ProfileName'] == profile]
    X_profile = profile_data['ProcessedText']
    y_profile = profile_data['Sentiment']

    # Convert text to numeric features
    X_profile_tfidf = vectorizer.transform(X_profile)

    # Encode labels
    y_profile_encoded = label_encoder.transform(y_profile)

    # Check the number of samples and classes before attempting to split
    if len(profile_data) < 2:
        print(f"Skipping profile {profile} due to insufficient samples.")
        continue

    # Check if there are multiple classes for the profile
    unique_classes = np.unique(y_profile_encoded)
    if len(unique_classes) < 2:
        print(f"Skipping profile {profile} due to only one class.")
        continue

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X_profile_tfidf, y_profile_encoded, test_size=0.2,
                                                        random_state=42)

    # Train the hybrid model
    hybrid_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_hybrid = hybrid_model.predict(X_test)

    # Store the classification report
    report = classification_report(y_test, y_pred_hybrid, zero_division=0)
    results[profile] = report

# Print results for each profile
print("\nClassification Reports by User Profile:")
for profile, report in results.items():
    print(f"\nProfile: {profile}")
    print(report)

# Calculate execution time
execution_time = time.time() - start_time
print(f"\nExecution Time (seconds): {execution_time:.2f}s")
