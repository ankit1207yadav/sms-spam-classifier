import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = str(text).lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

print("Loading data...")
df = pd.read_csv('spam.csv', encoding='latin-1')

# Data cleaning
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
df = df.rename(columns={'v1': 'target', 'v2': 'text'})

# Label encoding
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Remove duplicates
df = df.drop_duplicates(keep='first')

print("Preprocessing messages (this might take a minute)...")
df['transformed_text'] = df['text'].apply(transform_text)

print("Loading vectorizer...")
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

print("Vectorizing messages...")
X = tfidf.transform(df['transformed_text']).toarray()
y = df['target'].values

print("Training MultinomialNB model...")
model = MultinomialNB()
model.fit(X, y)

print("Saving model to model.pkl...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training complete! Model saved successfully.")
