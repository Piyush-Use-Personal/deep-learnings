# code/metadata_extractor.py

import json

def extract_metadata(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        "upload_date": data.get("upload_date"),
        "like_count": data.get("like_count"),
        "dislike_count": data.get("dislike_count"),
        "view_count": data.get("view_count"),
        "average_rating": data.get("average_rating")
    }
