for language in $(ls /cluster/work/projects/nn9851k/mariiaf/xlmr-toks/diachronic/)
    do
    for period in 2011_2015 2020_2021 2024_2025
        do
        sbatch --output=/cluster/work/projects/nn9851k/mariiaf/diachronic/logs/subst-xlmr-${language}_${period}-%j.out  get_substitutions.sh \
        --output-dir /cluster/work/projects/nn9851k/mariiaf/xlmr-toks/diachronic \
        --data-dir /cluster/work/projects/nn9851k/mariiaf/xlmr-toks/diachronic/ \
        --period $period \
        --max-batch-size 64 \
        --language $language \
        --model-name FacebookAI/xlm-roberta-base \
        --cache-dir /cluster/work/projects/nn9851k/hf_cache/
        done
    done