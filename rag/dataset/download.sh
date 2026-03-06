#!/bin/bash

mkdir wikilinks

for i in {0..9}; do
  curl -O "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/wiki-links/data-0000$i-of-00010.gz"
  gzip -d "data-0000$i-of-00010.gz"
  mv "data-0000$i-of-00010" wikilinks/
done

