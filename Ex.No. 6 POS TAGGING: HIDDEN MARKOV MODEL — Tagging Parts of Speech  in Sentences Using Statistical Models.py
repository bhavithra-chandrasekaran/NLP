import nltk
from nltk.tag import hmm
train_data = [
[("I", "PRP"), ("read", "VBP"), ("books", "NNS")],
[("She", "PRP"), ("reads", "VBZ"), ("daily", "RB")],
[("They", "PRP"), ("play", "VBP"), ("cricket", "NN")],
[("He", "PRP"), ("plays", "VBZ"), ("rice", "NN")],
]
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)
test_sentence = "He plays cricket".split()
tagged = tagger.tag(test_sentence)
print("Tagged Sentence:")
print(tagged)
