import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
def download_nltk_resources():
    try:
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        return True
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return False

if not download_nltk_resources():
    print("Failed to download required NLTK data. Exiting...")
    exit(1)

# Initialize NLP components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# Load or create sample dataset
try:
    df = pd.read_csv('fake_news_dataset.csv')  # Replace with your dataset
except FileNotFoundError:
    print("Dataset not found, creating sample data...")
    data = {
        'text': [
            'This is a real news article about important events',
            'This is completely fake information',
            'The government announced new policies today',
            'Celebrity spotted with aliens in secret location',
            'Scientists confirm climate change findings',
            'Moon landing was faked according to new theory'
        ],
        'label': [0, 1, 0, 1, 0, 1]  # 0=real, 1=fake
    }
    df = pd.DataFrame(data)

# Preprocess text
df['processed_text'] = df['text'].apply(preprocess_text)

# Create and fit vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=list(stop_words),
    ngram_range=(1, 2)
)
vectorizer.fit(df['processed_text'])

# Save vectorizer
os.makedirs('models', exist_ok=True)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Vectorizer successfully trained and saved to models/vectorizer.pkl")

# Optional: Train and save a simple classifier
from sklearn.linear_model import LogisticRegression

X = vectorizer.transform(df['processed_text'])
model = LogisticRegression()
model.fit(X, df['label'])

with open('models/fake_news_model.h5', 'wb') as f:
    pickle.dump(model, f)

print("Model successfully trained and saved to models/fake_news_model.h5")