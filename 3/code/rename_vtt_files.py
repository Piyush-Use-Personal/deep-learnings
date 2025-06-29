import os

def rename_vtt_files(vtt_dir):
    for filename in os.listdir(vtt_dir):
        if filename.endswith(".en.vtt"):
            old_path = os.path.join(vtt_dir, filename)
            new_filename = filename.replace(".en.vtt", ".vtt")
            new_path = os.path.join(vtt_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")

if __name__ == "__main__":
    rename_vtt_files("../downloads")
