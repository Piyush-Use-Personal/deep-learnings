import os
import csv
from vtt_to_text import convert_vtt_to_text
from keyphrase_extractor import extract_keyphrases
from classifier import classify_document
import json

def extract_metadata(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        "upload_date": data.get("upload_date"),
        "like_count": data.get("like_count"),
        "dislike_count": data.get("dislike_count"),
        "view_count": data.get("view_count"),
        "average_rating": data.get("average_rating")
    }

def process_all_documents(vtt_dir, text_dir, output_csv):
    records = []
    os.makedirs(text_dir, exist_ok=True)

    for filename in os.listdir(vtt_dir):
        if filename.endswith('.vtt'):
            vtt_path = os.path.join(vtt_dir, filename)
            base_name = filename.replace('.vtt', '')
            text_path = os.path.join(text_dir, base_name + '.txt')
            json_path = os.path.join(vtt_dir, base_name + '.info.json')

            if not os.path.exists(json_path):
                print(f"Skipping {filename} - no JSON found")
                continue

            convert_vtt_to_text(vtt_path, text_path)
            keyphrases = extract_keyphrases(text_path)
            domain, weights = classify_document(keyphrases)
            metadata = extract_metadata(json_path)

            record = {
                "file_name": filename,
                "domain": domain,
                **metadata,
                **weights
            }
            records.append(record)

    if records:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"✅ Final index saved to {output_csv}")
    else:
        print("⚠️ No records processed.")

# Run the pipeline
if __name__ == "__main__":
    process_all_documents("../downloads", "../text_files", "../index/final_index.csv")
