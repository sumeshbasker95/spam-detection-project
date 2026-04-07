# ============================================================
#  FINAL train_model.py (90.10% ACCURACY - 30K FEATURES)
# ============================================================
import pandas as pd
import re
import nltk
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# --- SETUP ---
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

def fast_preprocess(text: str) -> str:
    text = str(text).lower()
    text = EMAIL_RE.sub("emailaddr", text)
    text = URL_RE.sub("url", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return " ".join(w for w in text.split() if w not in STOP_WORDS and len(w) > 2)

def main():
    print("[INFO] Loading and Unifying Datasets...")
    df_email = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')[['label', 'text']]
    df_email['label'] = df_email['label'].map({'ham': 0, 'spam': 1})
    
    df_phish = pd.read_csv('malicious_phish.csv')
    df_phish['label'] = df_phish['type'].apply(lambda x: 0 if x == 'benign' else 1)
    df_phish = df_phish.rename(columns={'url': 'text'})[['label', 'text']]

    df = pd.concat([df_email, df_phish], ignore_index=True).dropna()
    df_spam = df[df['label'] == 1]
    df_ham = df[df['label'] == 0]
    
    # Balanced Sampling for 8GB RAM Stability
    min_count = min(len(df_spam), len(df_ham))
    df = pd.concat([df_spam.sample(min_count, random_state=42), 
                    df_ham.sample(min_count, random_state=42)]).sample(frac=1, random_state=42)

    print(f"[INFO] Preprocessing {len(df)} samples...")
    df["clean"] = df["text"].map(fast_preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )

    # 30,000 features for better word-combination detection
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 3), sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    
    # Alpha = 0.00001 (Extremely strict on Spam tokens)
    model = MultinomialNB(alpha=0.00001)
    model.fit(X_train_vec, y_train)

    with open("model.pkl", "wb") as f: pickle.dump(model, f)
    with open("tfidf.pkl", "wb") as f: pickle.dump(tfidf, f)
    
    y_pred = model.predict(tfidf.transform(X_test))
    print(f"\n✅ Validated Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    
    # Exporting Chart for Viva Documentation
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix: Spam Detection")
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    main()