# 📧 Spam Email Detector

A simple machine learning project that classifies emails as **spam** or **not spam** using natural language processing (NLP) techniques and logistic regression.

---

## 📁 Project Structure

- `email.csv` → Dataset file containing real email text labeled as **spam** or **ham**
- `emaildetect.py` → Python script that preprocesses the data, trains the model, and makes predictions

---

## 🧠 How It Works

1. **Preprocessing**  
   - Clean text using `re` (remove special characters, lowercase)
   - Remove stopwords and apply stemming using **NLTK**

2. **Feature Extraction**  
   - Use `TfidfVectorizer` to convert text into numerical form

3. **Model Training**  
   - Use `LogisticRegression` from **scikit-learn** to train the classifier

4. **Prediction**  
   - Make predictions on new emails using a trained model

---

## 🛠️ Libraries Used

- **pandas** → For reading and manipulating the dataset  
- **re** → Regular expressions for text cleaning  
- **nltk** → Natural Language Toolkit for text preprocessing (stopwords, stemming)  
- **scikit-learn** → For model training, vectorization, and evaluation

---

## ✅ How to Run

1. Install required libraries:
   ```bash
   pip install pandas nltk scikit-learn

