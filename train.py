import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the training dataset
data = pd.read_csv("intents.csv")

X = data["question"]
y = data["category"]

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train a machine learning text classifier
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
print("\nCategories:", list(model.classes_))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
joblib.dump(model, "intent_model.joblib")
print("\nModel saved as intent_model.joblib")
