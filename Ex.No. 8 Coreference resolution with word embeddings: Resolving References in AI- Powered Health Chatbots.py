import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = "I have severe back pain. It gets worse at night."
mentions = ["back pain"]
pronouns = {"it", "this", "that", "they", "he", "she", "these", "those"}
# Split text into sentences and compute TF-IDF representations
sentences = re.split(r"(?<=[.!?])\s+", text.strip())
corpus = mentions + sentences
vectors = TfidfVectorizer().fit_transform(corpus).toarray()
mention_vec = vectors[0]
sentence_vec = vectors[1:]
resolved = []
for i, sentence in enumerate(sentences):
    words = []
    for token in sentence.split():
        # Check if word is a pronoun and resolution applies to subsequent sentences
        if re.sub(r"[^\w]", "", token).lower() in pronouns and i > 0:
            score = cosine_similarity([sentence_vec[i - 1]], [mention_vec])[0][0]
            words.append(mentions[0] if score >= 0 else token)
        else:
            words.append(token)
    resolved.append(" ".join(words))
print("Original Text:")
print(text)
print("\nDetected Mention Candidates:")
print("-", mentions[0])
print("\nResolved Text:")
print(" ".join(resolved))
