#! /bin/bash

while read p;
do
    echo ${p}
    sbatch diachronic_mining.slurm ${p} 2011 2015 1000000
done < ${1}

