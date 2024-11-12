import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load or create dataset
data = {
    'text': [
        "I feel anxious and nervous all the time.",
        "I often feel sad and hopeless.",
        "I can't focus, my mind is always racing.",
        "I feel a strong urge to check things multiple times.",
        "I have no energy and lose interest in things."
    ],
    'label': ["anxiety", "depression", "ADHD", "OCD", "depression"]
}

df = pd.DataFrame(data)
# Split dataset
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# User input prediction
while True:
    user_text = input("Enter a description of how you feel (or type 'exit' to quit): ")
    if user_text.lower() == 'exit':
        break
    user_text_tfidf = vectorizer.transform([user_text])
    predicted_label = model.predict(user_text_tfidf)
    print("Predicted Mental Health Issue:", predicted_label[0])
