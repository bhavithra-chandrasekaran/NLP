import nltk
from nltk.corpus import conll2000

# Download the corpus
nltk.download("conll2000")

# Define a unigram chunker
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = []
        for sent in train_sents:
            iob_tags = nltk.chunk.tree2conlltags(sent)
            train_data.append([(pos, chunk) for word, pos, chunk in iob_tags])
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for word, pos in sentence]
        predicted_tags = self.tagger.tag(pos_tags)
        conll_tags = [
            (word, pos, chunk)
            for ((word, pos), (pos2, chunk))
            in zip(sentence, predicted_tags)
        ]
        return nltk.chunk.conlltags2tree(conll_tags)

# Load training and testing data
train_sents = conll2000.chunked_sents(
    "train.txt", chunk_types=["NP"]
)[:200]
test_sents = conll2000.chunked_sents(
    "test.txt", chunk_types=["NP"]
)[:50]

# Train the chunker
chunker = UnigramChunker(train_sents)

# Evaluate the chunker using the recommended accuracy method
score = chunker.accuracy(test_sents)

# New POS-tagged sentence
sample_sentence = [
    ("Google", "NNP"),
    ("opened", "VBD"),
    ("a", "DT"),
    ("new", "JJ"),
    ("office", "NN")
]

# Parse the new sentence
parsed_sentence = chunker.parse(sample_sentence)

print("Evaluation:")
print(score)
print("\nParsed Sentence:")
print(parsed_sentence)
