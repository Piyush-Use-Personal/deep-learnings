import re
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download the required NLTK model
nltk.download('averaged_perceptron_tagger')


# Load negation cue list (partial example)
negation_cues = set(['no', 'not', 'never', "can't", "don't", "isn't", "wasn't", "won't", "couldn't", "wouldn't", "aren't", "didn't", "doesn't"])

# Sample tweet dataset
data = {
    'tweet': [
        "I am not happy with the service",
        "This product isn't bad",
        "No one cares about your opinion",
        "The movie was great and amazing",
        "I don't think it was good",
        "Never going to use this again"
    ],
    'label': ['negative', 'positive', 'neutral', 'positive', 'negative', 'negative']
}
df = pd.DataFrame(data)

# --- Preprocessing and Negation Handling ---
def handle_negation(tweet):
    tokens = word_tokenize(tweet.lower())
    tagged = pos_tag(tokens)
    
    new_tokens = []
    negate = False
    window = 0
    max_window = 5

    for i, (word, tag) in enumerate(tagged):
        if word in negation_cues:
            negate = True
            window = 0
            continue
        if negate:
            if window < max_window and tag[0] in ['J', 'V', 'R', 'N']:  # Adjectives, Verbs, Adverbs, Nouns
                new_tokens.append(word + '_NEG')
                window += 1
            else:
                new_tokens.append(word)
                negate = False
        else:
            new_tokens.append(word)
    return " ".join(new_tokens)

df['processed_tweet'] = df['tweet'].apply(handle_negation)

# --- Feature Extraction ---
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['processed_tweet'])
y = df['label']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Classifiers ---
models = {
    'SVM': LinearSVC(),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier()
}

# --- Evaluation ---
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro')
    }

# --- Plotting ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
fig, ax = plt.subplots()
x = np.arange(len(metrics))
width = 0.25

for i, (model_name, scores) in enumerate(results.items()):
    ax.bar(x + i * width, [scores[m] for m in metrics], width=width, label=model_name)

ax.set_ylabel('Score')
ax.set_title('Model Comparison with Negation Handling')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.show()
