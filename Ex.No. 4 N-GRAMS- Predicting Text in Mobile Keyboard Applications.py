import nltk
from collections import defaultdict, Counter
from nltk.util import trigrams

def preprocess_text(input_text: str) -> list:
    """Tokenize and convert input string into lowercase words."""
    return input_text.lower().split()

def build_trigram_model(words: list) -> defaultdict:
    """Construct a trigram context-frequency dictionary model."""
    trigram_model = defaultdict(list)
    for w1, w2, w3 in trigrams(words):
        trigram_model[(w1, w2)].append(w3)
    for context_tuple in trigram_model:
        trigram_model[context_tuple] = Counter(trigram_model[context_tuple])
    return trigram_model

def predict_next_word(model: defaultdict, target_context: tuple) -> str:
    """Fetch the most probable next word for a given word pair context."""
    return model[target_context].most_common(1)[0][0]

def main():
    text = "I am going to college I am going to park I am going to market"
    words = preprocess_text(text)
    model = build_trigram_model(words)

    context = ("going", "to")
    prediction = predict_next_word(model, context)
    print("Context:", context)
    print("Predicted next word:", prediction)

if __name__ == "__main__":
    main()
