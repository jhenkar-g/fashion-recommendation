"""Helper utilities"""
import os, json
from pathlib import Path

def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, filepath):
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png')):
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'**/*{ext}'))
    return sorted(image_files)
