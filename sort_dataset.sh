#!/bin/bash

# Create a new directory to store subfolders
mkdir -p sorted_files

# Loop through each file in the current directory
for file in braille_dataset/*; do
    # Check if the file is a regular file
    if [[ -f "$file" ]]; then
        # Get the first character of the file name
        first_char=$(basename "$file" | cut -c1)

        # Create a subfolder if it doesn't exist
        mkdir -p "sorted_files/$first_char"

        # Copy the file into its respective subfolder
        cp "$file" "sorted_files/$first_char"
    fi
done

echo "Files sorted successfully."