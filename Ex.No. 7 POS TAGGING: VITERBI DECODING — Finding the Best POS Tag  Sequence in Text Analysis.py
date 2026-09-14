import nltk
from nltk.probability import DictionaryConditionalProbDist, DictionaryProbDist
from nltk.tag import hmm
# Define state space and vocabulary
states = ["N", "V"]
symbols = ["stocks", "rise"]
# Define initial probabilities
priors = DictionaryProbDist({"N": 0.6, "V": 0.4})
# Define state transition probabilities
transitions = DictionaryConditionalProbDist(
{
"N": DictionaryProbDist({"N": 0.3, "V": 0.7}),
"V": DictionaryProbDist({"N": 0.8, "V": 0.2}),
}
)
# Define emission probabilities
outputs = DictionaryConditionalProbDist(
{
"N": DictionaryProbDist({"stocks": 0.8, "rise": 0.2}),
"V": DictionaryProbDist({"stocks": 0.1, "rise": 0.9}),
}
)
# Initialize Hidden Markov Model Tagger
model = hmm.HiddenMarkovModelTagger(
symbols, states, transitions, outputs, priors
)
# Sequence prediction and probability calculation

24

observations = ["stocks", "rise"]
best_sequence = model.best_path(observations)
tagged_sequence = list(zip(observations, best_sequence))
probability = model.probability(tagged_sequence)
print("Best POS sequence:", best_sequence)
print("Probability:", round(probability, 4))
