import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from sklearn.metrics import confusion_matrix

nltk.download('wordnet')
nltk.download('punkt')

plt.style.use('ggplot')

class GWSPVisualFramework:
    def __init__(self):
        self.user_profiles = {}
        self.wordnet = wn
        self.search_results_db = {
            "apple size iPhone": [
                "iPhone 13 dimensions and size",
                "Apple iPhone measurement guide",
                "iPhone screen sizes comparison",
                "Physical dimensions of latest iPhones",
                "How big is the new iPhone?"
            ],
            "apple size fruit": [
                "Apple fruit size chart",
                "Measuring apple sizes for harvest",
                "Standard sizes for apple varieties",
                "How to measure apple diameter",
                "Apple size categories for orchards"
            ]
        }
        self.performance_metrics = {
            'UserA': {'precision': [], 'recall': [], 'accuracy': []},
            'UserB': {'precision': [], 'recall': [], 'accuracy': []}
        }
        
    def create_user_profile(self, user_id, basic_info=None, domains=None):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'basic_info': basic_info or {},
                'domains': defaultdict(lambda: {'terms': defaultdict(int), 'weight': 0}),
                'search_history': [],
                'dynamic_terms': defaultdict(int),
                'click_behavior': defaultdict(int)
            }
            if domains:
                for domain in domains:
                    self.user_profiles[user_id]['domains'][domain]['weight'] = 1
        return self.user_profiles[user_id]
    
    def process_query(self, user_id, query, domain=None):
        tokens = nltk.word_tokenize(query.lower())
        stems = [self._get_stem(token) for token in tokens]
        term_weights = {stem: stems.count(stem) for stem in set(stems)}
        
        combined_query = query
        if domain:
            combined_query = f"{domain} {query}"
            domain_stem = self._get_stem(domain)
            term_weights[domain_stem] = term_weights.get(domain_stem, 0) + 1
        
        return {
            'original_query': query,
            'domain': domain,
            'combined_query': combined_query,
            'stems': stems,
            'term_weights': term_weights,
            'user_id': user_id
        }
    
    def optimize_query(self, processed_query):
        user_id = processed_query['user_id']
        stems = processed_query['stems']
        domain = processed_query['domain']
        
        synonyms = {}
        for stem in stems:
            synsets = self.wordnet.synsets(stem)
            if synsets:
                synonyms[stem] = [lemma.name() for syn in synsets for lemma in syn.lemmas()]
        
        related_terms = []
        if user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
            
            if domain:
                domain_stem = self._get_stem(domain)
                if domain_stem in user_profile['domains']:
                    domain_terms = user_profile['domains'][domain_stem]['terms']
                    related_terms.extend([(term, weight) for term, weight in domain_terms.items()])
            
            related_terms.extend([(term, weight) for term, weight in user_profile['dynamic_terms'].items()])
        
        related_terms.sort(key=lambda x: x[1], reverse=True)
        
        optimized_terms = set(stems)
        for stem in stems:
            if stem in synonyms:
                optimized_terms.update(synonyms[stem][:2])
        
        optimized_terms.update([term for term, _ in related_terms[:3]])
        
        if domain:
            optimized_terms.add(self._get_stem(domain))
        
        optimized_query = " AND ".join([f"({term} OR {' OR '.join(synonyms.get(term, []))})" 
                                      if term in synonyms else term 
                                      for term in optimized_terms])
        
        return {
            'optimized_query': optimized_query,
            'synonyms': synonyms,
            'related_terms': related_terms,
            'original_query': processed_query['original_query'],
            'domain': domain
        }
    
    def execute_search(self, optimized_query, user_id):
        """Simulate search execution with personalized results"""
        key = optimized_query['original_query'] + " " + (optimized_query['domain'] or "")
        results = self.search_results_db.get(key, [])
        
        # Store performance metrics (simulated)
        precision = np.random.uniform(0.85, 0.99)
        recall = np.random.uniform(0.83, 0.97)
        accuracy = np.random.uniform(0.87, 0.98)
        
        self.performance_metrics[user_id]['precision'].append(precision)
        self.performance_metrics[user_id]['recall'].append(recall)
        self.performance_metrics[user_id]['accuracy'].append(accuracy)
        
        return {
            'results': results,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            }
        }
    
    def update_user_profile(self, user_id, query_data, clicked_results=None):
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        stems = query_data['stems']
        domain = query_data['domain']
        
        profile['search_history'].append({
            'query': query_data['original_query'],
            'domain': domain,
            'timestamp': pd.Timestamp.now()
        })
        
        if domain:
            domain_stem = self._get_stem(domain)
            if domain_stem not in profile['domains']:
                profile['domains'][domain_stem] = {'terms': defaultdict(int), 'weight': 0}
            
            profile['domains'][domain_stem]['weight'] += 1
            
            for stem in stems:
                profile['domains'][domain_stem]['terms'][stem] += 1
        
        for stem in stems:
            profile['dynamic_terms'][stem] += 1
        
        if clicked_results:
            for result in clicked_results:
                result_stems = [self._get_stem(token) for token in nltk.word_tokenize(result.lower())]
                for stem in result_stems:
                    if domain:
                        domain_stem = self._get_stem(domain)
                        profile['domains'][domain_stem]['terms'][stem] += 2
                    profile['dynamic_terms'][stem] += 2
                    profile['click_behavior'][stem] += 1
        
        return profile
    
    def _get_stem(self, word):
        return word.lower()
    
    def calculate_domain_weights(self, user_id):
        if user_id not in self.user_profiles:
            return None
        
        profile = self.user_profiles[user_id]
        for domain, data in profile['domains'].items():
            if data['terms']:
                total_weight = sum(data['terms'].values())
                data['weight'] = total_weight / len(data['terms'])
        
        return profile['domains']
    
    def visualize_user_profile(self, user_id):
        """Generate visualizations for user profile"""
        if user_id not in self.user_profiles:
            return None
        
        profile = self.user_profiles[user_id]
        
        # Create figure with subplots
        plt.figure(figsize=(18, 12))
        plt.suptitle(f"User Profile Visualization - {user_id}", fontsize=16)
        
        # Plot 1: Domain Weights
        plt.subplot(2, 2, 1)
        domains = [(d, data['weight']) for d, data in profile['domains'].items()]
        domains.sort(key=lambda x: x[1], reverse=True)
        if domains:
            df_domains = pd.DataFrame(domains, columns=['Domain', 'Weight'])
            sns.barplot(x='Weight', y='Domain', data=df_domains, palette='viridis')
            plt.title('Domain Weights')
            plt.xlabel('Weight')
            plt.ylabel('Domain')
        
        # Plot 2: Term Cloud
        plt.subplot(2, 2, 2)
        all_terms = {}
        for domain, data in profile['domains'].items():
            for term, weight in data['terms'].items():
                all_terms[term] = all_terms.get(term, 0) + weight
        
        if all_terms:
            wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(all_terms)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Term Word Cloud')
        
        # Plot 3: Click Behavior
        plt.subplot(2, 2, 3)
        if profile['click_behavior']:
            clicks = sorted(profile['click_behavior'].items(), key=lambda x: x[1], reverse=True)[:10]
            df_clicks = pd.DataFrame(clicks, columns=['Term', 'Clicks'])
            sns.barplot(x='Clicks', y='Term', data=df_clicks, palette='rocket')
            plt.title('Top Clicked Terms')
            plt.xlabel('Number of Clicks')
            plt.ylabel('Term')
        
        # Plot 4: Search History Timeline
        plt.subplot(2, 2, 4)
        if profile['search_history']:
            history = pd.DataFrame(profile['search_history'])
            history['timestamp'] = pd.to_datetime(history['timestamp'])
            history['count'] = 1
            history = history.set_index('timestamp').resample('D').count()
            
            if not history.empty:
                history['count'].plot(kind='line', marker='o')
                plt.title('Search Activity Over Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Searches')
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'user_profile_{user_id}.png')
        plt.show()
    
    def visualize_performance_metrics(self):
        """Compare performance metrics between users"""
        plt.figure(figsize=(12, 6))
        
        metrics = ['precision', 'recall', 'accuracy']
        users = list(self.performance_metrics.keys())
        
        # Create dataframe for metrics
        data = []
        for user in users:
            for metric in metrics:
                avg = np.mean(self.performance_metrics[user][metric]) * 100
                data.append({'User': user, 'Metric': metric.capitalize(), 'Value': avg})
        
        df = pd.DataFrame(data)
        
        # Plot grouped bar chart
        sns.barplot(x='Metric', y='Value', hue='User', data=df, palette='Set2')
        plt.title('Average Performance Metrics Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        plt.legend(title='User')
        
        # Add value labels
        for p in plt.gca().patches:
            plt.gca().annotate(f"{p.get_height():.1f}%", 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='center', 
                              xytext=(0, 5), 
                              textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('performance_metrics_comparison.png')
        plt.show()
    
    def visualize_query_optimization(self, original_query, optimized_query):
        """Visualize query expansion process"""
        plt.figure(figsize=(10, 4))
        
        # Extract components
        original_terms = set(nltk.word_tokenize(original_query.lower()))
        optimized_terms = set()
        
        # Parse optimized query to get all terms
        for part in optimized_query.split('AND'):
            terms = part.replace('(', '').replace(')', '').replace('OR', '').split()
            optimized_terms.update(terms)
        
        # Create Venn diagram data
        from matplotlib_venn import venn2
        venn2([original_terms, optimized_terms], 
              set_labels=('Original Terms', 'Optimized Terms'),
              set_colors=('skyblue', 'lightgreen'),
              alpha=0.7)
        
        plt.title('Query Term Expansion Visualization')
        plt.savefig('query_expansion.png')
        plt.show()
    
    def visualize_search_results(self, results, user_id, query):
        """Visualize personalized search results"""
        plt.figure(figsize=(10, 6))
        
        # Simulate relevance scores (for visualization)
        relevance = np.linspace(100, 70, len(results))
        
        # Create dataframe
        df = pd.DataFrame({
            'Result': [r[:50] + '...' for r in results],
            'Relevance': relevance,
            'Position': range(1, len(results)+1)
        })
        
        # Plot results
        sns.barplot(x='Relevance', y='Result', data=df, palette='Blues_d')
        plt.title(f"Personalized Search Results for User {user_id}\nQuery: '{query}'")
        plt.xlabel('Relevance Score')
        plt.ylabel('Search Result (truncated)')
        
        # Add position numbers
        for i, (_, row) in enumerate(df.iterrows()):
            plt.text(row['Relevance']-5, i, f"#{row['Position']}", 
                    ha='right', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'search_results_{user_id}.png')
        plt.show()

# Demonstration with Visualizations
if __name__ == "__main__":
    # Initialize framework
    gwsp = GWSPVisualFramework()
    
    # Create user profiles
    user_a = "UserA"
    gwsp.create_user_profile(user_a, 
                           basic_info={"name": "Alice", "interests": ["technology"]},
                           domains=["iPhone", "smartphones"])
    
    user_b = "UserB"
    gwsp.create_user_profile(user_b,
                           basic_info={"name": "Bob", "interests": ["fruits", "agriculture"]},
                           domains=["fruit", "apple varieties"])
    
    # Simulate multiple searches to build profiles
    queries = [
        (user_a, "apple size", "iPhone"),
        (user_a, "latest model", "iPhone"),
        (user_a, "screen dimensions", None),
        (user_b, "apple size", "fruit"),
        (user_b, "harvest season", "apple varieties"),
        (user_b, "nutritional value", "fruit")
    ]
    
    for user, query, domain in queries:
        processed = gwsp.process_query(user, query, domain)
        optimized = gwsp.optimize_query(processed)
        search_results = gwsp.execute_search(optimized, user)
        
        # Simulate clicking top 2 results
        clicked = search_results['results'][:2]
        gwsp.update_user_profile(user, processed, clicked)
        
        # Visualize search results for the first query of each user
        if (user == user_a and query == "apple size") or (user == user_b and query == "apple size"):
            gwsp.visualize_search_results(search_results['results'], user, query)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # User profile visualizations
    gwsp.visualize_user_profile(user_a)
    gwsp.visualize_user_profile(user_b)
    
    # Performance metrics comparison
    gwsp.visualize_performance_metrics()
    
    # Query optimization visualization for User A
    processed = gwsp.process_query(user_a, "apple size", "iPhone")
    optimized = gwsp.optimize_query(processed)
    gwsp.visualize_query_optimization("apple size iPhone", optimized['optimized_query'])
    
    print("Visualizations saved as PNG files.")