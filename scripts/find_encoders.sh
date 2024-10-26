#!/bin/bash

# Get all FFmpeg encoders
encoders=$(ffmpeg -encoders 2>/dev/null | grep -oP "(?<=\s)[\w_]+(?=\s{2,}E)")

echo "Searching for encoders supporting gbrp16le pixel format..."

# Loop through each encoder and check if it supports gbrp16le
for encoder in $encoders; do
    # Get the supported pixel formats for each encoder
    ffmpeg -h encoder=$encoder 2>/dev/null | grep -q "gbrp16le"
    
    # If the encoder supports gbrp16le, print its name
    if [ $? -eq 0 ]; then
        echo "$encoder supports gbrp16le"
    fi
done
