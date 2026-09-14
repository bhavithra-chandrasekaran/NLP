import nltk
from nltk import word_tokenize, pos_tag
# Download required resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
# Resume text
text = """
Experienced Python developer with strong machine learning skills
and natural language processing knowledge
"""
# Tokenization
tokens = word_tokenize(text)
# POS tagging
tagged = pos_tag(tokens)
# Chunk grammar
grammar = r"""
SKILL: {<JJ>*<NNP|NN>+<NN|NNS>*}
"""



# Create parser and parse the tagged words
parser = nltk.RegexpParser(grammar)
chunk_tree = parser.parse(tagged)
# Display POS tags
print("POS Tags:")
print(tagged)
# Display chunk tree
print("\nChunk Tree:")
print(chunk_tree)
# Extract and display skill phrases
print("\nExtracted Skill Phrases:")
for subtree in chunk_tree.subtrees():
    if subtree.label() == "SKILL":
        phrase = " ".join(word for word, tag in subtree.leaves())
        print("-", phrase)
