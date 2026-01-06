#!/bin/bash
#SBATCH --job-name=lchange
#SBATCH --account=nn10029k
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

# Merges token embeddings into one-file-per-initial

echo "--Node: $(hostname)"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"


mkdir -p subsets/${LANG}/t5_embeddings/${period}
apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 torch_merge.py ${@}
