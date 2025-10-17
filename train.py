import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

DATA_PATH = "emails.csv"

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Put your dataset at {DATA_PATH} (CSV with 'label','text').")

df = pd.read_csv(DATA_PATH, encoding='latin-1')
# keep only label and text if dataset has extra columns
if 'label' not in df.columns:
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1':'label','v2':'text'})
    else:
        raise SystemExit("Dataset must have 'label' and 'text' columns.")

df = df[['label','text']].dropna()
df['label'] = df['label'].map(lambda x: 1 if str(x).strip().lower().startswith('s') else 0)

X = df['text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)),
    ('clf', LogisticRegression(max_iter=1000))
])

print("Training...")
pipeline.fit(X_train, y_train)

print("Evaluating...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham','spam']))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

MODEL_PATH = "saved_model.joblib"
joblib.dump(pipeline, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
