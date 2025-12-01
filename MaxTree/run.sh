#!/usr/bin/env bash

# Exit on error
set -e

# Check if argument provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input-image>"
    exit 1
fi

INPUT="$1"
BASENAME=$(basename "$INPUT")
NAME="${BASENAME%.*}"
PGM_FILE="${NAME}.pgm"
OUTPUT_JPG="out.jpg"

# Convert input image to PGM
ffmpeg -y -i "$INPUT" "$PGM_FILE"

# Run maxtree on the PGM
./maxtree "$PGM_FILE" a 0 1 2 3 4 5 6 nogui

# Check if out.pgm was created
if [ ! -f "out.pgm" ]; then
    echo "Error: out.pgm not created by maxtree"
    exit 1
fi

# Convert out.pgm to JPG
ffmpeg -y -i out.pgm "$OUTPUT_JPG"

echo "Done. Output written to: $OUTPUT_JPG"
