#!/bin/bash
#SBATCH --job-name=lchange
#SBATCH --account=nn10029k
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G


# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

echo "--Node: $(hostname)"
SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"

LANG=${1}  # Language to process
DATA=${2} # Where to find the datasets? (e.g., "subsets")

echo ${LANG}

periods=("2011_2015" "2020_2021")

for period in "${periods[@]}";
    do
        echo ${period}
        # We pass as arguments the path to the training corpus and (optionally) the number of CPU cores we want to use
        apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 src/align.py -e0 ${DATA}/${LANG}/word2vec/2024_2025.model -e1 ${DATA}/${LANG}/word2vec/${period}.model -c 10 -t languages/${LANG}/target_words.json &
    done

wait



