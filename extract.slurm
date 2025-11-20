#!/bin/bash
#SBATCH --job-name=lchange
#SBATCH --account=project_465002310
#SBATCH --partition=small
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


EBU_USER_PREFIX=/projappl/project_465001925/software/

module --quiet purge
module load LUMI

LANG=${1}  # Path to language in the catalogue
FIELD=${2} # What field to extract (ts, crawl_id, etc)

mkdir -p ${LANG}

echo ${LANG}
echo ${FIELD}

for el in /appl/local/openeurollm/training/catalogue/hplt/3.0/sorted/${LANG}/*.jsonl.zst
do
    echo ${el}
    time -p zstdcat ${el} | jq --arg FIELD $FIELD -r '"\(.[$FIELD])"' | gzip >> ${LANG}/${FIELD}.txt.gz
done

time -p zcat ${LANG}/${FIELD}.txt.gz | cut -d '-' -f 1 | sort | uniq -c | sort -nr > ${LANG}/${FIELD}_sorted.txt

