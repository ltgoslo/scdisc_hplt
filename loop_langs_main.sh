#! /bin/bash

while read p;
do
    echo ${p}
    sbatch diachronic_mining.slurm ${p} 2024 2025 1000000
done < ${1}

