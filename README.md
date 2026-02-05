# Diachronic HPLT datasets for semantic change discovery

This repository accompanies the paper `DHPLT: large-scale multilingual diachronic corpora and word representations for semantic change modelling` from LChange'26 workshop.

The DHPLT dataset is published [here](https://data.hplt-project.org/three/diachronic/).


## Usage Info
### Exploring HPLTv3 documents & stats
`src/extract.slurm <langauge> <info_field>` --- extracts information about the given language and the document field of interest. As part of the script, `scr/dia_distribution.py` is invoked to visualize the information. `src/extract_large.slurm` can be used for larger languages.

We look at the distribution of documents by the web crawl (`languages/<language>/crawl_id_sorted.png`) and plot it (`languages/<language>/crawl_id.png`). 

`languages/dia_lang_stats.tsv` shows document counts per year and diachronic time period. `languages/dia_stats.tsv` shows average document and segment (paragraph) length for each langauge and time period. `languages/languages2process.tsv` presents document 

### Sampling diachronic data
`src/diachronic_mining.slurm <language> <period_start> <period_end> <sample_size>` --- samples `sample_size`-many documents from the original [HPLT v3](https://data.hplt-project.org/three/) data in the given language, such that the crawl time stamps for these documents fall between `period_end` and `period_end`.

### Processing documents
`src/tokenize_data.py <input_path> <tokenizer> <output_dir>` --- for the given language and time period (as provided by `input_path`) tokenizes the documents and splits the text into segments (paragraphs) at newline character.

### Selecting target terms
`vocabulary` --- contains scripts, requirements, and information for target term selection. More detailed instructions are listed under  `vocabulary/prepare_vocab__README.md`

### Filtering segments
`src/filter_datasets.slurm <language> <data_dir>` --- filters out segments that don't contain target terms. 

### Static word embeddings
`src/extract_text.slurm <language> <data_dir>` --- splits tokenized documents in the given language into segments (paragraphs) and extracts the text of each one 

`src/train_word2vec.slurm <language> <data_model_dir>` --- trains a word2vec model for the given language located under the `data_model_dir` directory.

`src/align_word2vec.slurm  <language> <data_model_dir>` --- for the given language and directory for data and word2vec models, the script aligns models from earlier periods to the model from the last period.

### Contextualized embeddings
`src/get_many_embeddings.sh` --- obtains T5 embeddings for multiple languages across three time period

`src/get_xlrm_embeddings.sh` --- obtains XLMR embeddings for multiple languages across three time period

### Masked-token substitutes 
`src/get_many_substitutes.sh` --- obtains XLRM substitutes for multiple languages across three time period
