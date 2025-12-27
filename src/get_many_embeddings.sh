for language in als_Latn ast_Latn bel_Cyrl bos_Latn bul_Cyrl cat_Latn ces_Latn dan_Latn ekk_Latn ell_Grek fin_Latn fra_Latn hrv_Latn hun_Latn ind_Latn ita_Latn lit_Latn lvs_Latn mkd_Cyrl nld_Latn nob_Latn pol_Latn por_Latn ron_Latn rus_Cyrl slk_Latn slv_Latn spa_Latn swe_Latn tur_Latn ukr_Cyrl vie_Latn
    do
    for period in 2011_2015 2020_2021 2024_2025
        do
        sbatch --output=/cluster/work/projects/nn9851k/mariiaf/diachronic/logs/t5emb_$language-%j.out get_contextualized_embeddings.sh \
            --data-dir /cluster/work/projects/nn9851k/mariiaf/diachronic/ \
            --embeddings-dir /cluster/work/projects/nn9851k/mariiaf/diachronic \
            --max-batch-size 400 \
            --language $language \
            --model-name HPLT/hplt_t5_base_3_0_$language \
            --cache_dir  /cluster/work/projects/nn9851k/hf_cache/ \
            --period $period
        done
    done