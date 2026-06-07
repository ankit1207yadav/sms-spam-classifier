# Email/SMS Spam Classifier

An end-to-end Machine Learning web application built using **Streamlit** and **Scikit-Learn** to classify SMS and email messages as **Spam** or **Ham (Not Spam)**.

## Project Structure
*   `app.py`: Streamlit frontend application.
*   `train_model.py`: Script to train the `MultinomialNB` classifier using `spam.csv` and `vectorizer.pkl`.
*   `model.pkl`: Trained Multinomial Naive Bayes model.
*   `vectorizer.pkl`: Fitted TF-IDF Vectorizer with 3,000 max features.
*   `spam.csv`: SMS Spam Collection dataset used for training.
*   `requirements.txt`: Python package dependencies.
*   `.gitignore`: Git ignore patterns.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ankit1207yadav/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources:**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

## Running the Web Application
Start the Streamlit development server:
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.
