from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text, stemming=False):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    if stemming:
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]
    
    return ' '.join(words)

def document_similarity(doc1, doc2, stemming=False):
    """Calculates cosine similarity between two documents after preprocessing."""
    preprocessed1 = preprocess(doc1, stemming)
    preprocessed2 = preprocess(doc2, stemming)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed1, preprocessed2])
    
    vec1 = tfidf_matrix[0].toarray().flatten()
    vec2 = tfidf_matrix[1].toarray().flatten()
    
    if not vec1.any() or not vec2.any():
        return 0.0  
    
    cosine_sim = 1 - cosine(vec1, vec2)
    return cosine_sim

doc1 = "Virat playing cricket"
doc2 = "Subhash started playing cricket by seeing Virat's centuries"

similarity = document_similarity(doc1, doc2, stemming=False)

print(f"Cosine Similarity: {similarity:.2f}")

if similarity > 0.5:
    print("The documents are similar.")
else:
    print("The documents are not similar.")
