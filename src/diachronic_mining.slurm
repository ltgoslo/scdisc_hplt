#!/bin/bash
#SBATCH --job-name=lchange
#SBATCH --account=nn10029k
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G

# Extracts documents crawled in a specific time period
# Saves them in a single jsonl.zst file

LANG=${1}  # Language
PERIOD0=${2} # The first year for extraction
PERIOD1=${3} # The last year for extraction
DOCS=${4} # How many documents to sample (1000000 or 500000)
N=8 # Number of files processed in parallel

# For smaller languages, one can use local ${SCRATCH} of the compute node
# (max 3.5T)
mkdir -p /cluster/work/projects/nn9851k/tmp/${LANG}
mkdir -p diachronic/${LANG}

echo ${LANG}
echo ${PERIOD0}
echo ${PERIOD1}
echo ${DOCS}
echo ${N}

for el in /nird/datalake/NS8112K/public/three/sorted/${LANG}/*.jsonl.zst
do
    (
    echo ${el}
    zstdcat ${el} | jq -c --arg s ${PERIOD0} --arg e ${PERIOD1} 'select(.ts | . >= $s and . <= $e + "z") | {id: .id, ts: .ts, text: .text, doc_scores: .doc_scores}' | zstd > /cluster/work/projects/nn9851k/tmp/${LANG}/${PERIOD0}_${PERIOD1}_$(basename ${el}).jsonl.zst
    sleep $(( (RANDOM % 3) + 1))
    ) &
    # Allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # Now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi
done

# No more jobs to be started but wait for pending jobs
# (all need to be finished)
wait

echo "Sampling..."

# This is for really large languages like English, Russian or Chinese:

for el in /cluster/work/projects/nn9851k/tmp/${LANG}/${PERIOD0}_${PERIOD1}_*.jsonl.zst
do
    (
    zstdcat ${el} | shuf -n 10000 | zstd > /cluster/work/projects/nn9851k/tmp/${LANG}/sampled_${PERIOD0}_${PERIOD1}_$(basename ${el})
    sleep $(( (RANDOM % 3) + 1))
    ) &
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        wait -n
    fi
done
wait

zstdcat /cluster/work/projects/nn9851k/tmp/${LANG}/sampled_${PERIOD0}_${PERIOD1}_*.zst | shuf -n ${DOCS} | zstd > diachronic/${LANG}/${PERIOD0}_${PERIOD1}.jsonl.zst

# This is for lesser languages:

# zstdcat /cluster/work/projects/nn9851k/tmp/${LANG}/${PERIOD0}_${PERIOD1}_*.zst | shuf -n ${DOCS} | zstd > diachronic/${LANG}/${PERIOD0}_${PERIOD1}.jsonl.zst

rm /cluster/work/projects/nn9851k/tmp/${LANG}/${PERIOD0}_${PERIOD1}_*.zst

echo "All done"
