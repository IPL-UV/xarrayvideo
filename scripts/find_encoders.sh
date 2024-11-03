#!/bin/bash

#Usage Examples
#./find_encoders.sh yuv444p
#./find_encoders.sh gbrp16le
#./find_encoders.sh 16le

# Set pixel format from the first argument, or default to "gbrp16le" if none is provided
PIXEL_FORMAT="${1:-gbrp16le}"

# Get all FFmpeg encoders (extracting only the encoder names from the second column)
encoders=$(ffmpeg -encoders 2>/dev/null | awk '{if ($1 ~ /^V/ || $1 ~ /^A/ || $1 ~ /^S/) print $2}')

echo "Searching for encoders supporting $PIXEL_FORMAT pixel format:"

# Loop through each encoder and check if it supports the specified pixel format
for encoder in $encoders; do
    # Get the supported pixel formats for each encoder
    ffmpeg -h encoder=$encoder 2>/dev/null | grep -q "$PIXEL_FORMAT"
    
    # If the encoder supports the specified pixel format, print its name
    if [ $? -eq 0 ]; then
        echo "$encoder : $PIXEL_FORMAT"
    fi
done
