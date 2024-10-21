#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <directory> <\"name1 name2 ... nameN\">"
  exit 1
fi
# bash delete.sh "save/" "Attention preTCN"

target_dir=$1

dir_names=($2)

# Iterate over each directory name in the array
for dir_name in "${dir_names[@]}"; do
  echo "Searching for directories named '$dir_name' in $target_dir ..."

  # Use find to locate and delete directories matching the names
  find "$target_dir" -type d -name "$dir_name" -exec rm -rf {} \;

  echo "Deleted directories named '$dir_name' in $target_dir."
done

echo "Deletion process completed."