import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords
nltk.download("stopwords")

# Initialize stemmer and stopwords set
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Load dataset and fix columns
df = pd.read_csv("email.csv", encoding="latin-1")[["Category", "Message"]]
df.columns = ["label", "message"]

# Map labels to binary (ham = 0, spam = 1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ✅ Remove any rows with missing values
df.dropna(subset=["label", "message"], inplace=True)

# Preprocessing function with regex, stopwords, and stemming
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)     # Remove special chars
    text = text.lower()                        # Lowercase
    words = text.split()                       # Tokenize
    filtered = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(filtered)

# Apply preprocessing
df["cleaned_message"] = df["message"].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=3000)
x = vectorizer.fit_transform(df["cleaned_message"])
y = df["label"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict a custom email
def predict_email(email_text):
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "spam" if prediction[0] == 1 else "not spam"

# Example usage
#email = "Congratulations! You've won a free iPhone. Click here to claim now"
email = "Dear Aryan verma,Your dream career begins with one decision, and the clock is ticking.Bennett University, powered by the legacy of  The Times Group, offers world-class UG & PG programs that blend academic excellence with industry exposure.Application deadline 31st May 2025.Click here to Apply : http://applications.bennett.edu.in/verify/YTozOntzOjY6InNvdXJjZSI7czo1OiJlbWFpbCI7czoxNDoidXNlcl91bmlxdWVfaWQiO3M6MzY6ImM3Y2ExOTI5LTU4NjAtNGNhZS1hYTdiLWM4Yzc5ZTkxOTNhZSI7czo3OiJzZW50X2VtIjtzOjIxOiJhYmhpdmVyOTAyN0BnbWFpbC5jb20iO30="

print(f"\n📧 Email: {email}\n🧠 Prediction: {predict_email(email)}")
