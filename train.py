import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------------------------------
# CONFIG
DATA_PATH = "emails.csv"   # rename your Kaggle CSV to this
MODEL_PATH = "saved_model.joblib"
# -----------------------------------------------------

# 1. Load dataset
print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Ensure correct columns
if not {'text', 'spam'}.issubset(df.columns):
    raise ValueError("CSV must contain columns: 'text' and 'spam'")

# 2. Clean data
df = df.dropna(subset=['text', 'spam'])
df['text'] = df['text'].astype(str)
df['spam'] = df['spam'].astype(int)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['spam'], test_size=0.2, random_state=42, stratify=df['spam']
)

# 4. Build pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(max_iter=2000, C=2, solver='lbfgs'))
])

# 5. Train
print("Training model...")
pipeline.fit(X_train, y_train)

# 6. Evaluate
print("\nEvaluating on test set...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham','spam']))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save model
joblib.dump(pipeline, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
