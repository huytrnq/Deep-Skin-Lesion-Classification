#!/bin/bash

# Root directory containing class folders
ROOT_DIR=/Users/huytrq/Workspace/UdG/CAD/Data/Challenge1/val
OUTPUT_FILE=val.txt

# Ensure the root directory exists
if [ ! -d "$ROOT_DIR" ]; then
  echo "Error: Directory '$ROOT_DIR' does not exist."
  exit 1
fi

# Initialize the output file
echo "Generating file list in '$OUTPUT_FILE'..."
> "$OUTPUT_FILE"

# Get class folders and assign class IDs (starting from 0)
CLASS_ID=0
for CLASS_FOLDER in "$ROOT_DIR"/*; do
  if [ -d "$CLASS_FOLDER" ]; then
    # Iterate through image files in the class folder
    for IMAGE_FILE in "$CLASS_FOLDER"/*.{png,jpg,jpeg,bmp,tiff}; do
      # Check if file exists (to handle cases with no matching files)
      if [ -f "$IMAGE_FILE" ]; then
        # Write the path and class ID to the output file
        echo "$IMAGE_FILE $CLASS_ID" >> "$OUTPUT_FILE"
      fi
    done
    CLASS_ID=$((CLASS_ID + 1)) # Increment class ID
  fi
done

echo "File list created successfully in '$OUTPUT_FILE'."