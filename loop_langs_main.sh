#! /bin/bash

while read p;
do
    echo ${p}
    # sbatch diachronic_mining.slurm ${p} 2024 2025 1000000
    sbatch filter_datasets.slurm ${p} /cluster/work/projects/nn9851k/mariiaf/diachronic/ /cluster/work/projects/nn9851k/corpora/diachronic/
done < ${1}

