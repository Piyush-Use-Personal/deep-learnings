# -----------------------------------------------
# File: visualize.py
# Purpose: Generate charts from final_index.csv
# -----------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the final CSV index
df = pd.read_csv("../index/final_index.csv")

# Clean date field (in case it's not datetime yet)
df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
df['year'] = df['upload_date'].dt.year

# --------------------------
# 📊 1. Bar Chart: Domain Counts
# --------------------------
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='domain', order=df['domain'].value_counts().index, palette="pastel")
plt.title("📊 Video Count per Computer Science Domain")
plt.xticks(rotation=45)
plt.xlabel("CS Domain")
plt.ylabel("Number of Videos")
plt.tight_layout()
plt.show()

# --------------------------
# 🕒 2. Line Chart: Year-wise Trends
# --------------------------
trend = df.groupby(['year', 'domain']).size().unstack().fillna(0)
trend.plot(marker='o', figsize=(12, 6))
plt.title("🕒 CS Domain Trends Over Years")
plt.xlabel("Year")
plt.ylabel("Video Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# ☁️ 3. Word Cloud: Topic Domains
# --------------------------
text = " ".join(df['domain'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("☁️ Word Cloud of Classified CS Domains")
plt.tight_layout()
plt.show()
