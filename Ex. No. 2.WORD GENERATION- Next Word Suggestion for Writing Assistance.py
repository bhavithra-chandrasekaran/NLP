import nltk
from nltk import bigrams, word_tokenize
from collections import defaultdict, Counter
import random
# Sample text data
sentences = [
"please find attached the report",
"please let me know your feedback",
"kindly find the updated document",
"please find the meeting agenda",
"let me know the schedule"
]
model = defaultdict(list)
for sent in sentences:
words = sent.split()
for i in range(len(words) - 1):
model[words[i]].append(words[i + 1])
next_word_model = {}
for word, next_words in model.items():
next_word_model[word] = Counter(next_words).most_common(1)[0][0]
input_word = "find"
print("Next word suggestion for", input_word, ":", next_word_model.get(input_word, "No
suggestion"))
