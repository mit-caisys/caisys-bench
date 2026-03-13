#!/bin/bash

mkdir -p data/videos

for i in {01..20}
do
  echo "---------------------------"
  echo "Processing chunk $i..."


  ZIP_NAME="videos_chunked_${i}.zip"
  curl -L "https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/${ZIP_NAME}" -o "$ZIP_NAME"


  python3 filter_video_mme.py "$ZIP_NAME"


  rm "$ZIP_NAME"

  echo "Finished chunk $i."
done

echo "All chunks processed."