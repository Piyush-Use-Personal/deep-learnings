# -------------------------------
# cs_trend_analysis.py
# -------------------------------

import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# -------------------------------
# Step 1: Prepare Dataset
# -------------------------------

data = {
    'Topic': ['Machine Learning'] * 3 + ['Cybersecurity'] * 3 + ['Artificial Intelligence'] * 3 +
             ['Data Science'] * 3 + ['Blockchain'] * 3,
    'Title': [
        "Introduction to Machine Learning", "Supervised vs Unsupervised Learning", "Building ML Models",
        "Cybersecurity Basics", "How Hackers Attack", "Network Security Explained",
        "What is Artificial Intelligence?", "AI vs Human Intelligence", "Neural Networks Explained",
        "Data Science Tutorial", "Analyzing Data with Python", "Predictive Modeling Basics",
        "What is Blockchain?", "How Bitcoin Works", "Decentralized Finance Explained"
    ],
    'Description': [
        "Learn the basics of ML including algorithms, datasets and models.",
        "Comparison of supervised and unsupervised learning methods.",
        "Hands-on tutorial for building machine learning models.",
        "Cybersecurity introduction for beginners and threat types.",
        "Explore common hacking techniques and how to prevent them.",
        "Explanation of network security and firewalls.",
        "Definition and scope of AI with real-world examples.",
        "Differences between artificial and human intelligence.",
        "Concepts behind neural networks and deep learning.",
        "Complete data science course covering pandas and numpy.",
        "Performing analysis using Python libraries and data cleaning.",
        "Predictive modeling using machine learning tools.",
        "Learn the fundamentals of blockchain technology.",
        "Understand Bitcoin and cryptocurrency mining.",
        "Explanation of DeFi and smart contracts."
    ],
    'PublishedDate': pd.date_range(start='2023-01-01', periods=15, freq='ME')  # 'ME' = Month End
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Text Preprocessing
# -------------------------------

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Combine title + description
df['Text'] = df['Title'] + ' ' + df['Description']

# Define stopwords
stop_words = set(stopwords.words('english'))

# Tokenize and clean
def clean_and_tokenize(text):
    tokens = nltk.word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

df['Tokens'] = df['Text'].apply(clean_and_tokenize)

# -------------------------------
# Step 3: Keyword Frequency Analysis
# -------------------------------

# Flatten token list
all_tokens = [word for tokens in df['Tokens'] for word in tokens]
word_freq = Counter(all_tokens)

# Show top 10 keywords
print("\n🔝 Top 10 Keywords:")
for word, freq in word_freq.most_common(10):
    print(f"{word}: {freq}")

# -------------------------------
# Step 4: Word Cloud Visualization
# -------------------------------

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("📊 WordCloud of Frequent Keywords")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 5: Bar Chart of Top Keywords
# -------------------------------

top_words = word_freq.most_common(10)
words, freqs = zip(*top_words)

plt.figure(figsize=(8, 5))
plt.bar(words, freqs, color='skyblue')
plt.title("🔠 Top 10 Frequent Keywords")
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
