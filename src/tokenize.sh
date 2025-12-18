#!/bin/bash
#SBATCH --job-name=stat_dia
#SBATCH --account=nn10029k
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=123
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=normal


echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"

SIF="/cluster/projects/nn9851k/containers/pytorch2.7_cu2.9_py3.12_amd_nlpl.sif"

srun apptainer exec -B /cluster/projects/:/cluster/projects/,/cluster/work/projects/:/cluster/work/projects/ $SIF python3 tokenize_data.py ${@}

