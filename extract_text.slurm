#!/bin/bash
#SBATCH --job-name=lchange
#SBATCH --account=nn10029k
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=1G

# Extracts only text from diachronic datasets, one segment per line
# Replaces punctuation with whitespaces, removes leading and trailing tabs and whitespaces

echo "--Node: $(hostname)"

LANG=${1}  # Language to process
DATA=${2} # Where to find the tokenized datasets? (e.g., subsets/)

echo ${LANG}

periods=("2011_2015" "2020_2021" "2024_2025")

for period in "${periods[@]}";
    do
        echo ${period}
        zstdcat ${DATA}/${LANG}/${period}.jsonl.zst | jq -r .text | sed 's/[[:punct:]]/ /g' | sed 's/^[[:blank:]]*//;s/[[:blank:]]*$//' | sed '/^$/d' | zstd > ${DATA}/${LANG}/${period}.txt.zst &
    done
wait
