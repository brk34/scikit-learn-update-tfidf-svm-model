import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Örnek veri seti
data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
X, y = data.data, data.target

# TfidfVectorizer ve SVM Pipeline oluşturma
vectorizer = TfidfVectorizer(stop_words='english')
svm_model = LinearSVC()

# Modeli ve vectorizer'ı birleştir
model_pipeline = make_pipeline(vectorizer, svm_model)

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli eğit
model_pipeline.fit(X_train, y_train)

# Model ve Vectorizer'ı kaydet
with open('vectorizer2.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('svm2.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

print("Model ve vectorizer başarıyla kaydedildi.")
