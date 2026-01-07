for language in $(ls /cluster/work/projects/nn9851k/mariiaf/xlmr-toks/diachronic/)
    do
    echo $language
    for period in 2011_2015 2020_2021 2024_2025
        do
        sbatch --output=/cluster/work/projects/nn9851k/mariiaf/diachronic/logs/xlmr_${language}_${period}-%j.out get_contextualized_embeddings.sh \
            --data-dir /cluster/work/projects/nn9851k/mariiaf/xlmr-toks/diachronic/ \
            --embeddings-dir /cluster/work/projects/nn9851k/mariiaf/xlmr-toks/diachronic \
            --max-batch-size 64 \
            --language $language \
            --model-name FacebookAI/xlm-roberta-base \
            --cache_dir  /cluster/work/projects/nn9851k/hf_cache/ \
            --period $period
        done
    done