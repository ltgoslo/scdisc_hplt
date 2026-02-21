#!/bin/bash
#SBATCH --job-name=lchange
#SBATCH --account=nn10029k
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6G

# Merges all substitutions for one word in one dictionary

echo "--Node: $(hostname)"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"

LANG=${1}  # Language to process
DATA=${2} # Where to find the substitutions? (e.g., diachronic/)

REPR="bert_substitutions"

echo ${LANG}

periods=("2011_2015" "2020_2021" "2024_2025")

for period in "${periods[@]}";
    do
        echo ${period}
        mkdir -p subsets/${LANG}/${REPR}/${period}
        apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 src/subst_merge.py ${DATA}${LANG}/${period}__substitutions/ subsets/${LANG}/${REPR}/${period} &
    done
wait
