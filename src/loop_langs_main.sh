#! /bin/bash

while read p;
do
    echo ${p}
    # sbatch diachronic_mining.slurm ${p} 2024 2025 1000000
    # sbatch filter_datasets.slurm ${p} /cluster/work/projects/nn9851k/mariiaf/diachronic/
    # sbatch extract_text.slurm ${p} subsets
    # sbatch train_word2vec.slurm ${p} subsets
    # sbatch align_word2vec.slurm ${p} subsets
    sbatch merge_embeddings.slurm ${p} /cluster/work/projects/nn9851k/mariiaf/diachronic/
done < ${1}
