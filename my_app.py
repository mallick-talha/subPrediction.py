import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset for demonstration
data = {
    "text": ["The sky is blue", "Python is a programming language", "Mona Lisa is a famous painting"],
    "subject": ["Science", "Technology", "Art"]
}

# Convert data to DataFrame
import pandas as pd
df = pd.DataFrame(data)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Define target labels
y = df['subject']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Function to predict subject
def predict_subject(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Main function
def main():
    st.title("Subject Prediction App")
    st.write("This app predicts the subject of a given text.")

    # Input text
    text = st.text_area("Enter a sentence or text:")

    # Predict subject
    if st.button("Predict"):
        if text:
            subject = predict_subject(text)
            st.write(f"Predicted Subject: {subject}")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
