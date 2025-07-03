# Quora Insincere Question Classifier

A machine learning model that classifies questions as **sincere** or **insincere** using the [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification) dataset from Kaggle.

---

## Project Overview

This project uses traditional NLP techniques (TF-IDF) and a Logistic Regression model to identify whether a question contains toxic, hateful, or insincere content. The model is trained on real user-generated data from Quora.

---

## Technologies & Tools

- Python
- Jupyter Notebook
- Scikit-learn (`sklearn`)
- TF-IDF Vectorization
- Logistic Regression
- NLTK (for stopwords)
- Seaborn & Matplotlib (for plotting)

---

## Dataset

- **Source**: [Kaggle - Quora Insincere Questions Classification](https://www.kaggle.com/competitions/quora-insincere-questions-classification)
- **Columns used**:
  - `question_text` — The actual user question
  - `target` — 1 (insincere) or 0 (sincere)

---

1. **Preprocessing**:
   - Lowercasing
   - Removing punctuation/numbers
   - Removing stopwords (via NLTK)

2. **Feature Extraction**:
   - TF-IDF Vectorizer (Top 5000 words)

3. **Model**:
   - Logistic Regression with `class_weight='balanced'` to handle imbalanced data

4. **Evaluation**:
   - Accuracy
   - F1 Score
   - Confusion Matrix

5. **Interactive Prediction**:
   - You can enter your own question and the model will tell if it is sincere or not.

---

## Results

| Metric        | Score |
|---------------|-------|
| Accuracy      | 0.8743 |
| F1 Score      | 0.4566 |
| Confusion Matrix | Visualized using seaborn |

