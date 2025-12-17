#!/bin/bash
#SBATCH --job-name=lchange
#SBATCH --account=nn10029k
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=1G

# Filters diachronic datasets based on sets of target tokens

echo "--Node: $(hostname)"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"

LANG=${1}  # Language to process
DATA=${2} # Where to find the tokenized datasets? (e.g., /cluster/work/projects/nn9851k/mariiaf/diachronic/)

echo ${LANG}

periods=("2011_2015" "2020_2021" "2024_2025")

for period in "${periods[@]}";
    do
        echo ${period}
        apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 src/filter_non_target_segments_tokenized.py --language ${LANG} --period ${period} --data-dir ${DATA}  | zstd > ${DATA}${LANG}/${period}_tokens_filtered.jsonl.zst &
    done
wait