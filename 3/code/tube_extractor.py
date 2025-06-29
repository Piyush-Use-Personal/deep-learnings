# ---------------------------------------------
# File: tube_extractor.py
# Purpose: Download metadata (VTT + JSON) from YouTube videos for CS topics
# ---------------------------------------------

import os
import yt_dlp  # You must install this via: pip install yt-dlp

def download_youtube_data(query, download_path="../downloads/"):
    """
    Downloads subtitle (VTT) and metadata (JSON) for a YouTube search query.
    Only metadata/subtitles are downloaded (video is skipped).
    """

    # Create download folder if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Define yt_dlp options
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'{download_path}/video_%(id)s.%(ext)s',  # Safe filename: video_<id>.ext
        'writeautomaticsub': True,        # Download auto-generated subtitles
        'writeinfojson': True,            # Save metadata
        'subtitleslangs': ['en'],         # Only English
        'skip_download': True,            # Do NOT download video
        'quiet': False                    # Show progress
    }

    # Start the download process
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"\nSearching and downloading metadata for: {query}")
        try:
            ydl.download([f"ytsearch1:{query}"])  # 1 video per query
            print(f"Download complete for: {query}")
        except Exception as e:
            print(f"Error downloading {query}: {e}")

# ----------------------------
# Example usage 
# ----------------------------
if __name__ == "__main__":
    # List of computer science topic queries
    cs_topics = [
        "Machine Learning tutorial",
        "Computer Networks explained",
        "Cybersecurity basics",
        "Artificial Intelligence introduction",
        "Blockchain in computer science"
    ]

    for topic in cs_topics:
        download_youtube_data(topic)
