import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Sample dataset
data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
X, y = data.data, data.target

# Create TF-IDF Vectorizer and SVM Pipeline
vectorizer = TfidfVectorizer(stop_words='english')
svm_model = LinearSVC()

# Combine the model and vectorizer
model_pipeline = make_pipeline(vectorizer, svm_model)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model and vectorizer
with open('vectorizer2.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('svm2.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

print("Model and vectorizer have been successfully saved.")
