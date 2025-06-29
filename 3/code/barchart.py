import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../index/final_index.csv")
domain_counts = df['domain'].value_counts()

domain_counts.plot(kind='bar', title="Video Count per CS Domain", color='skyblue')
plt.xlabel("Domain")
plt.ylabel("Video Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
