import xgboost    
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Suppress KMeans warning for n_init
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

print("Starting eHTRUST Model Simulation...\n")

#### 1. Phase I: Elimination of Phishing Websites ####

print("Phase I: Eliminating Phishing Websites...")

initial_urls_data = []
for i in range(1, 51):
    url_no = f'U{i}'
    is_phishing = 0
    f1, f2, f4, f10 = 0, 0, 0, 0
    if i % 7 == 0:
        is_phishing = 1
        f1 = 1
        f10 = 1
    elif i % 11 == 0:
        is_phishing = 1
        f2 = 1
        f4 = 1

    initial_urls_data.append({
        'URL_No': url_no,
        'F1': f1, 'F2': f2, 'F3': 1, 'F4': f4, 'F5': 0, 'F6': 0,
        'F7': 0, 'F8': 0, 'F9': 0, 'F10': f10, 'F11': 0, 'F12': 1,
        'is_phishing': is_phishing
    })

initial_urls_df = pd.DataFrame(initial_urls_data)
X_phishing = initial_urls_df[[f'F{i}' for i in range(1, 13)]]
y_phishing = initial_urls_df['is_phishing']

dummy_phishing_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
dummy_phishing_model.fit(X_phishing, y_phishing)

initial_urls_df['predicted_is_phishing'] = dummy_phishing_model.predict(X_phishing)

legitimate_urls_df = initial_urls_df[initial_urls_df['predicted_is_phishing'] == 0].copy()
legitimate_urls_df = legitimate_urls_df.drop(columns=['is_phishing', 'predicted_is_phishing'])

print(f"Initial URLs: 50")
print(f"Legitimate URLs identified: {len(legitimate_urls_df)}")
print("Sample Legitimate URLs:\n", legitimate_urls_df['URL_No'].head())
print("-" * 50)

#### 2. Phase II: Dataset Preparation ####

print("\nPhase II: Preparing Dataset with User-Independent Factors...")

legit_url_nos = legitimate_urls_df['URL_No'].tolist()

if 'U6' in legit_url_nos and 'U7' in legit_url_nos:
    legit_url_nos = [url for url in legit_url_nos if url not in ['U6', 'U7']]
    legit_url_nos.insert(0, 'U6 & U7')

user_independent_factors_data = []
for url_no in legit_url_nos:
    if url_no == 'U6 & U7':
        user_independent_factors_data.append({
            'URL_No': url_no,
            'Response_time': 1.33,
            'SSL_Certified': 1,
            'Reviewed': 0,
            'Responsive': 1,
            'Date_Diff_days': 0,
            'Transparency': 1
        })
    else:
        user_independent_factors_data.append({
            'URL_No': url_no,
            'Response_time': round(np.random.uniform(0.08, 6.86), 2),
            'SSL_Certified': 1,
            'Reviewed': np.random.choice([0, 1], p=[0.9, 0.1]),
            'Responsive': np.random.choice([0, 1], p=[0.2, 0.8]),
            'Date_Diff_days': np.random.choice([-1, 0, 4, 5, 61]),
            'Transparency': np.random.choice([0, 1], p=[0.3, 0.7])
        })

dataset_phase2_df = pd.DataFrame(user_independent_factors_data)

print(f"Dataset prepared for {len(dataset_phase2_df)} legitimate URLs.")
print("Sample of prepared dataset:\n", dataset_phase2_df.head())
print("-" * 50)

#### 3. Phase III: Clustering of URLs ####

print("\nPhase III: Clustering URLs using K-Means...")

features_for_clustering = dataset_phase2_df[['Response_time', 'SSL_Certified', 'Reviewed', 'Responsive', 'Date_Diff_days', 'Transparency']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

kmeans = KMeans(n_clusters=3, init='random', max_iter=5, random_state=42, n_init=1)
dataset_phase2_df['Cluster'] = kmeans.fit_predict(scaled_features)

print("URLs clustered into 3 groups.")
print("Sample of clustered URLs:\n", dataset_phase2_df[['URL_No', 'Cluster']].head())
print("-" * 50)

#### 4. Trust Score Calculation and Labeling ####

print("\nCalculating Trust Scores and Labeling Clusters...")

df_with_scores = dataset_phase2_df.copy()
max_response_time_observed = df_with_scores['Response_time'].max()
df_with_scores['Response_time_Score'] = (1 - (df_with_scores['Response_time'] / max_response_time_observed)) * 100

df_with_scores['SSL_Certified_Score'] = df_with_scores['SSL_Certified'] * 100
df_with_scores['Reviewed_Score'] = df_with_scores['Reviewed'] * 100
df_with_scores['Responsive_Score'] = df_with_scores['Responsive'] * 100
df_with_scores['Transparency_Score'] = df_with_scores['Transparency'] * 100
df_with_scores['Date_Score'] = df_with_scores['Date_Diff_days'].apply(lambda x: 100 if x >= 0 else 0)

factor_scores_cols = [
    'Response_time_Score', 'SSL_Certified_Score', 'Reviewed_Score',
    'Responsive_Score', 'Date_Score', 'Transparency_Score'
]
df_with_scores['Trust_Score'] = df_with_scores[factor_scores_cols].mean(axis=1)

print("Sample URLs with calculated Trust Scores:\n", df_with_scores[['URL_No', 'Trust_Score', 'Cluster']].head())

mean_trust_scores_per_cluster = df_with_scores.groupby('Cluster')['Trust_Score'].mean().reset_index()
mean_trust_scores_per_cluster.columns = ['Cluster', 'Mean_Trust_Score']

sorted_clusters_by_trust = mean_trust_scores_per_cluster.sort_values(by='Mean_Trust_Score', ascending=False)

cluster_labels_map = {}
if len(sorted_clusters_by_trust) == 3:
    cluster_labels_map[sorted_clusters_by_trust.iloc[0]['Cluster']] = 'Most Trustable'
    cluster_labels_map[sorted_clusters_by_trust.iloc[1]['Cluster']] = 'Trustable'
    cluster_labels_map[sorted_clusters_by_trust.iloc[2]['Cluster']] = 'Less Trustable'
elif len(sorted_clusters_by_trust) == 2:
    cluster_labels_map[sorted_clusters_by_trust.iloc[0]['Cluster']] = 'Most Trustable'
    cluster_labels_map[sorted_clusters_by_trust.iloc[1]['Cluster']] = 'Less Trustable'
else:
    cluster_labels_map[sorted_clusters_by_trust.iloc[0]['Cluster']] = 'Trustable'

df_with_scores['Category'] = df_with_scores['Cluster'].map(cluster_labels_map)

print("\nCluster-wise Mean Trust Scores and Labels:")
print(mean_trust_scores_per_cluster)
print("\nFinal Categorization of URLs (Sample):\n", df_with_scores[['URL_No', 'Cluster', 'Trust_Score', 'Category']].head())
print("-" * 50)
