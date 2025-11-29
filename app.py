import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.title("📩 SMS Spam Detector")

# ------------------------------
# Load dataset
# ------------------------------
data = pd.read_csv("sms_spam_dataset.csv", encoding="latin-1")
data = data[['Label', 'Message']]
data.columns = ['Label', 'Message']
data['Label'] = data['Label'].str.strip().str.lower()

# Train/test split
X = data['Message']
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ------------------------------
# UI Input
# ------------------------------
msg = st.text_area("messages:")

if st.button("Check Spam"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_tfidf = vectorizer.transform([msg])
        pred = model.predict(msg_tfidf)[0]

        if pred == "spam":
            st.error("🚨 Result: **it is SPAM!**")
        else:
            st.success("✅ Result: **it is not SPAM**")
