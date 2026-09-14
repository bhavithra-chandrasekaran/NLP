import nltk
import re
from collections import Counter
# Step 1: Customer reviews
reviews = [
    "Food was cold and delivery was late",
    "Late delivery and the order was wrong",
    "Food taste was good but packaging was bad",
    "Delivery boy was rude and food was cold",
    "Wrong items delivered and very late service",
    "Food was cold and package was not good"
]
# Step 3: Remove stop words
stop_words = {"was", "and", "the", "but", "very", "is", "a"}
# Step 2, 3, 4: Convert to lowercase, tokenize, remove stop words
tokens = []
for review in reviews:
    words = re.findall(r'\b[a-zA-Z]+\b', review.lower())
    words = [w for w in words if w not in stop_words]
    tokens.extend(words)
# Step 5: Count word frequencies

freq = Counter(tokens)
# Step 6: Display top complaint words
print("Top Complaint Words:")
print(freq.most_common(10))
# Step 7: Identify common issues
print("\nCommon Issues:")
if freq["cold"] > 2:
    print("- Cold food")
if freq["late"] > 2:
    print("- Late delivery")
if freq["wrong"] > 1:
    print("- Wrong order")
if freq["bad"] > 1:
    print("- Bad packaging")
if freq["rude"] > 0:
    print("- Rude delivery behavior")
