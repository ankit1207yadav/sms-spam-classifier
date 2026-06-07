# 📩 Email/SMS Spam Classifier

An end-to-end Machine Learning web application built using **Streamlit** and **Scikit-Learn** to classify SMS and email messages as **Spam** or **Ham (Not Spam)**.

### 🔗 Live Demo: [Streamlit Web App](https://sms-spam-classifier-qvcybyewaqkzrbvwhf86qx.streamlit.app/)

---

## 🚀 Model Performance
The classifier is powered by a **Multinomial Naive Bayes (MultinomialNB)** algorithm trained on the SMS Spam Collection dataset, preprocessed with Porter Stemming and TF-IDF vectorization:
*   **Accuracy:** **97.03%**
*   **Precision:** **100.0%** (0 false positives on the test set, ensuring legitimate emails are not incorrectly flagged as spam)

---

## 🛠️ Project Structure
*   `app.py`: Streamlit web application.
*   `train_model.py`: Training script to preprocess the dataset, train the `MultinomialNB` model, and serialize it.
*   `model.pkl`: Fully fitted Naive Bayes classifier model.
*   `vectorizer.pkl`: Fitted TF-IDF Vectorizer with a vocabulary size of 3,000.
*   `spam.csv`: SMS Spam Collection training dataset.
*   `requirements.txt`: Python dependencies configuration.
*   `.gitignore`: List of untracked files/folders (such as the virtual environment `venv/`).

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ankit1207yadav/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources:**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   Open **http://localhost:8501** in your web browser.
