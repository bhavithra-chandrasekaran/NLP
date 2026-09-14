import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

def setup_nltk_resources():
    """Download required NLTK corpus datasets."""
    nltk.download("wordnet")
    nltk.download("omw-1.4")

def initialize_nlp_tools():
    """Instantiate stemming and lemmatization objects."""
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    return stemmer, lemmatizer

def format_table_row(word: str, stem: str, lemma: str) -> str:
    """Format single string output matching original tabbed spacing."""
    return f"{word}\t\t{stem}\t\t{lemma}"

def process_word_list(words: list):
    """Process input vocabulary list and output header with results."""
    setup_nltk_resources()
    stemmer, lemmatizer = initialize_nlp_tools()
    print("Word\t\tStem\t\tLemma")

    for word in words:
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word)
        formatted_line = format_table_row(word, stem, lemma)
        print(formatted_line)

if __name__ == "__main__":
    target_words = ["playing", "studies", "better", "running", "wolves"]
    process_word_list(target_words)
