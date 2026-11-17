# Practical 7: Text Analytics
# Tokenization, POS tagging, stop-word removal, stemming, lemmatization, TF-IDF.

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4")
]
for path, package in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)

document = """
Data science helps people analyze data. Students are learning analytics,
processing documents, and discovering useful information from data.
"""

tokens = word_tokenize(document)
print("Tokens:\n", tokens)

print("\nPOS tagging:\n", pos_tag(tokens))

stop_words = set(stopwords.words("english"))
filtered = [w for w in tokens if w.isalpha() and w.lower() not in stop_words]
print("\nAfter stop-word removal:\n", filtered)

stemmer = PorterStemmer()
print("\nStemming:\n", [stemmer.stem(w) for w in filtered])

lemmatizer = WordNetLemmatizer()
print("\nLemmatization:\n", [lemmatizer.lemmatize(w.lower()) for w in filtered])

documents = [
    "Data science analyzes useful information.",
    "Text analytics processes documents and words.",
    "Students learn data analysis and machine learning."
]
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(documents)
print("\nTF-IDF feature names:\n", vectorizer.get_feature_names_out())
print("\nTF-IDF matrix:\n", matrix.toarray())
print("\nPractical 7 completed.")
