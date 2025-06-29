import os
import re

def convert_vtt_to_text(input_vtt_path, output_text_path):
    with open(input_vtt_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    cleaned_lines = []
    for line in lines:
        if re.match(r'^\d\d:\d\d:\d\d\.\d\d\d', line):
            continue
        line = re.sub(r'<.*?>', '', line).strip()
        if line and line not in cleaned_lines:
            cleaned_lines.append(line)

    with open(output_text_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(cleaned_lines))
