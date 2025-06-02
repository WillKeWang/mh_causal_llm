#!/usr/bin/env bash
##############################################################################
#  list_structure_and_headers.sh
#  Recursively prints directory structure and CSV headers.
#  Usage:  ./list_structure_and_headers.sh /absolute/path/to/DATA_DIR
##############################################################################

set -euo pipefail

## ---- 1. Where is the data? -----------------------------------------------
DATA_DIR=${1:-$(pwd)}          # default = current directory if no arg passed
echo -e "\n=======  DATA DIRECTORY:  $DATA_DIR  =======\n"

## ---- 2. Show directory tree (uses only find & sed, no 'tree' dependency) --
echo "----------  Folder / file hierarchy  ----------"
find "$DATA_DIR" -print | sed "s|$DATA_DIR|.|" | sort
echo "----------  End hierarchy  ---------------------"

## ---- 3. Enumerate column headers in every CSV -----------------------------
echo -e "\n==========  CSV column summaries  ============\n"
find "$DATA_DIR" -type f -name '*.csv' | sort | while read -r csvfile; do
    relpath="${csvfile#$DATA_DIR/}"        # path relative to root dir
    echo "File: $relpath"
    # grab first line, split on commas, number each column
    head -n 1 "$csvfile" | tr ',' '\n' | nl -w2 -s': '
    echo                                                       # blank line
done
echo "===============  Done  ========================="
