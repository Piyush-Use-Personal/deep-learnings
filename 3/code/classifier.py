# code/classifier.py

from collections import defaultdict

CSO_DICT = {
    "tcp": "Computer Networks",
    "machine learning": "Artificial Intelligence",
    "operating system": "Computer Operating Systems",
    "javascript": "Internet",
    "web server": "Computer Security"
}

def classify_document(keyphrases):
    domain_freq = defaultdict(int)
    for phrase in keyphrases:
        domain = CSO_DICT.get(phrase.lower())
        if domain:
            domain_freq[domain] += 1

    total = sum(domain_freq.values())
    domain_weights = {k: v/total for k, v in domain_freq.items()} if total else {}
    final_domain = max(domain_weights, key=domain_weights.get, default="Unknown")

    return final_domain, domain_weights
