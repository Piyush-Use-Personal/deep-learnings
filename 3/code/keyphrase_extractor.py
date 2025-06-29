from rake_nltk import Rake
import nltk

nltk.download('stopwords')

def extract_keyphrases(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()
