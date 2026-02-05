for language in $(ls /cluster/work/projects/nn9851k/mariiaf/xlmr-toks/diachronic/)
    do
    echo $language
    sbatch filter_datasets.slurm ${language} /cluster/work/projects/nn9851k/mariiaf/xlmr-toks/diachronic/
    done