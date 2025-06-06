# Naive Bayes Classifier from Scratch

This project implements the Naive Bayes classification algorithm from scratch in Python using two real-world datasets: a **Golf Decision dataset** for binary classification (Bernoulli Naive Bayes) and a **Tweet Sentiment dataset** from the HuggingFace `tweet_eval` collection (Multinomial Naive Bayes). We also compare results with scikit-learn's built-in implementations.

---

## 📁 Datasets Used

- **Golf Dataset**: Local CSV file with categorical features determining if one should play golf.
- **Tweet Eval Dataset**: `emotion` subset from the HuggingFace `tweet_eval` dataset for multi-class emotion classification.

---

## 🔧 Project Structure

1. **Data Preprocessing**
   - Label encoding of golf dataset features.
   - Cleaning tweets using regex, stopword removal, and lowercasing.

2. **Custom Implementations**
   - `BernoulliNaiveBayes` for binary features.
   - `NaiveBayes` for multinomial features using Bag-of-Words vectorization.

3. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1 Score
   - Confusion Matrix

4. **Comparison with scikit-learn**
   - `BernoulliNB` for Golf dataset
   - `MultinomialNB` for Tweet dataset

---

## 🧪 Evaluation Results
Bernoulli Naive Bayes (Golf Data) Metrics on Test Set:
Accuracy: 0.82
Precision: 0.52
Recall: 0.03
F1 Score: 0.06
Confusion Matrix:
[[1872   13]
 [ 401   14]]

> **Note:** Actual results may vary depending on software versions and the randomness in train/test splits.

---

## 🧠 Key Learnings

- Implementing Naive Bayes from scratch deepens understanding of conditional probabilities and Laplace smoothing.
- Bag-of-Words model is a simple yet powerful way to vectorize text.
- Comparing with scikit-learn helps validate correctness and measure performance.

---

## 📌 Future Improvements

- Use TF-IDF for better text representation.
- Extend the model to support Gaussian Naive Bayes for continuous data.
- Add unit tests and modularize code further.
