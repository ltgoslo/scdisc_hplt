source .venv/bin/activate
cd ..

select_languages=("als_Latn" "arb_Arab" "bos_Latn" "bul_Cyrl" "cat_Latn" "ces_Latn" "cmn_Hans" 
				  "dan_Latn" "deu_Latn" "ekk_Latn" "ell_Grek" "eng_Latn" "fin_Latn" "fra_Latn" 
				  "heb_Hebr" "hrv_Latn" "hun_Latn" "hye_Armn" "ind_Latn" "ita_Latn" "jpn_Jpan" 
				  "kat_Geor" "kor_Hang" "lit_Latn" "lvs_Latn" "mkd_Cyrl" "nld_Latn" "nob_Latn" 
				  "pol_Latn" "por_Latn" "ron_Latn" "rus_Cyrl" "slk_Latn" "slv_Latn" "spa_Latn" 
				  "swe_Latn" "tam_Taml" "tha_Thai" "tur_Latn" "ukr_Cyrl" "vie_Latn")
periods=("2011_2015" "2020_2021" "2024_2025")

cd "../../data"
for language in "${select_languages[@]}"; do
	mkdir -p "${language}"
	cd "${language}"
	echo "LANGUAGE: ${language}"
	# pwd
	for period in "${periods[@]}"; do
		if [ ! -f "${period}.jsonl.zst" ]; then
			echo "Downloading '${language}/${period}.jsonl.zst' ..."
			wget --quiet "https://data.hplt-project.org/three/diachronic/${language}/${period}.jsonl.zst"
		fi
	done
	cd "../scdisc_hplt/vocabulary"
	python prepare_vocabulary.py --language "${language}"\
								 --data-dir "../../data"\
								 --inter-period-min-freq 10\
								 --intra-period-min-freq 30\
								 --not-compute-full-word-freqs\
								 --not-run-pos-tagging
	# pwd
	# cd "../data/${language}"
	# for period in "${periods[@]}"; do
	# 	rm "${period}.jsonl.zst"
	# done
	# pwd
	# cd ..
	cd "../data"
	echo "------------------------------------------------------"
done

