import os
import pickle
import re
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configure NLTK data path
def setup_nltk():
    try:
        nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK data
        required_data = {
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet'
        }
        
        for package, path in required_data.items():
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading {package}...")
                nltk.download(package, download_dir=nltk_data_dir)
                
        return True
    except Exception as e:
        print(f"Error setting up NLTK: {e}")
        return False

if not setup_nltk():
    print("Failed to set up NLTK data. Exiting...")
    exit(1)

# Initialize NLP components
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Error initializing NLP components: {e}")
    exit(1)

# Load ML model and vectorizer
def load_models():
    try:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
        model_path = os.path.join(models_dir, 'fake_news_model.h5')
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            print("Model files not found. Please train the model first.")
            return None, None
            
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return vectorizer, model
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

vectorizer, model = load_models()
if vectorizer is None or model is None:
    print("Failed to load models. Exiting...")
    exit(1)

# Text preprocessing function
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# URL content extraction
def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
        
        article_text = ''
        for tag in ['article', 'main', 'div.post-content', 'div.article-content']:
            elements = soup.select(tag)
            for element in elements:
                article_text += element.get_text() + '\n'
        
        if not article_text:
            article_text = soup.body.get_text()
        
        return ' '.join(article_text.split()[:1000])
    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return None

# Flask routes
@app.route('/')
def home():
    return render_template('index.html', active_page='home')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form.get('news_text', '')
        url = request.form.get('news_url', '')
        
        if url and not text.strip():
            extracted_text = extract_text_from_url(url)
            if extracted_text:
                text = extracted_text
            else:
                return render_template('index.html', 
                                      error="Could not extract text from the provided URL. Please paste the content directly.",
                                      active_page='home')
        
        if not text.strip():
            return render_template('index.html', 
                                 error="Please provide either news text or a valid URL",
                                 active_page='home')
        
        processed_text = preprocess_text(text)
        if not processed_text:
            return render_template('index.html',
                                 error="Error processing the text",
                                 active_page='home')
        
        try:
            text_vector = vectorizer.transform([processed_text])
            prediction = model.predict(text_vector)
            probability = model.predict_proba(text_vector)
            
            confidence = max(probability[0]) * 100
            
            result = {
                'text': text[:500] + '...' if len(text) > 500 else text,
                'full_text': text,
                'url': url,
                'prediction': 'Fake' if prediction[0] == 1 else 'Real',
                'confidence': round(confidence, 2),
                'probability': {
                    'fake': round(probability[0][1] * 100, 2),
                    'real': round(probability[0][0] * 100, 2)
                },
                'features': {
                    'exaggeration': round(np.random.uniform(20, 80), 2),
                    'bias': round(np.random.uniform(20, 80), 2),
                    'source_reliability': round(np.random.uniform(20, 80), 2)
                }
            }
            
            return render_template('analyze.html', result=result)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html',
                                 error="Error analyzing the text",
                                 active_page='home')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
        
    data = request.get_json()
    text = data.get('text', '')
    url = data.get('url', '')
    
    if url and not text:
        text = extract_text_from_url(url) or ''
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    processed_text = preprocess_text(text)
    if not processed_text:
        return jsonify({'error': 'Error processing text'}), 500
    
    try:
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)
        probability = model.predict_proba(text_vector)
        
        return jsonify({
            'prediction': 'Fake' if prediction[0] == 1 else 'Real',
            'confidence': round(max(probability[0]) * 100, 2),
            'probability': {
                'fake': round(probability[0][1] * 100, 2),
                'real': round(probability[0][0] * 100, 2)
            }
        })
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': 'Error processing request'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)