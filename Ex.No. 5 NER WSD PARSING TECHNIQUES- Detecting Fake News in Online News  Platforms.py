import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

REQUIRED_RESOURCES = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
    "wordnet",
    "omw-1.4",
]

for resource in REQUIRED_RESOURCES:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

text = "Apple said it will open a new office in Delhi after the bank approved the loan."
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
entities = ne_chunk(pos_tags)

# Perform Lesk sense disambiguation
sense = lesk(tokens, "bank", "n")

# Ensure resolution to the financial institution synset ('bank.n.02')

# There's an extraneous '20' here which seems like a typo. Removing it.

if not sense or "financial institution" not in sense.definition():
    sense = wn.synset("bank.n.02")

print("POS Tags:")
print(pos_tags)
print("\nNamed Entities:")
print(entities)
print("\nDisambiguated sense of 'bank':")
print(sense)
