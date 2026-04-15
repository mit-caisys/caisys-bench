import sys
import os
import zipfile
import shutil
from datasets import load_dataset

LIMIT_SHORT = 100
LIMIT_MEDIUM = 100
BASE_DIR = "./data/videos"
SHORT_DIR = os.path.join(BASE_DIR, "short")
MEDIUM_DIR = os.path.join(BASE_DIR, "medium")

if len(sys.argv) < 2:
    print("Usage: python3 filter_and_extract.py <zip_file>")
    sys.exit(1)

zip_path = sys.argv[1]

print("Loading dataset metadata...")
try:
    ds = load_dataset("lmms-lab/Video-MME", split="test")
    duration_map = {f"{row['videoID']}.mp4": row['duration'] for row in ds}
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

for d in [SHORT_DIR, MEDIUM_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

count_short = len([f for f in os.listdir(SHORT_DIR) if f.endswith(".mp4")])
count_medium = len([f for f in os.listdir(MEDIUM_DIR) if f.endswith(".mp4")])

print(f"Status: Have {count_short}/{LIMIT_SHORT} Short, {count_medium}/{LIMIT_MEDIUM} Medium.")

if count_short >= LIMIT_SHORT and count_medium >= LIMIT_MEDIUM:
    print("Goal reached! Stopping.")
    sys.exit(2) 

# 3. Extract Loop
print(f"Scanning {zip_path}...")
try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_info in z.infolist():
            # Stop if limits met
            if count_short >= LIMIT_SHORT and count_medium >= LIMIT_MEDIUM:
                break

            filename = os.path.basename(file_info.filename)
            if not filename or filename not in duration_map:
                continue

            duration = duration_map[filename]
            target_folder = None

            # Determine where to put the file (and if we need it)
            if duration == "short" and count_short < LIMIT_SHORT:
                target_folder = SHORT_DIR
                count_short += 1
            elif duration == "medium" and count_medium < LIMIT_MEDIUM:
                target_folder = MEDIUM_DIR
                count_medium += 1
            
            # Extract if we assigned a folder
            if target_folder:
                source = z.open(file_info)
                target_path = os.path.join(target_folder, filename)
                
                try:
                    with open(target_path, "wb") as f:
                        shutil.copyfileobj(source, f)
                except OSError as e:
                    if e.errno == 28: # Disk full error
                        print("CRITICAL: Disk full! Deleting partial file and exiting.")
                        try: os.remove(target_path) 
                        except: pass
                        sys.exit(1)
                    raise e

    if count_short >= LIMIT_SHORT and count_medium >= LIMIT_MEDIUM:
        sys.exit(2)

except zipfile.BadZipFile:
    print(f"Error: {zip_path} is corrupted.")