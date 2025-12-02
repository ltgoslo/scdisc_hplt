#!/bin/bash
#SBATCH --job-name=lchange
#SBATCH --account=nn10029k
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24

# Extracts documents crawled in a specific time period
# Saves them in a single jsonl.zst file

LANG=${1}  # Language
PERIOD0=${2} # The first year for extraction
PERIOD1=${3} # The last year for extraction
N=16 # Number of files processed in parallel

mkdir -p ${LANG}

echo ${LANG}
echo ${PERIOD0}
echo ${PERIOD1}
echo ${N}


for el in /cluster/projects/nn10029k/HPLT3/${LANG}/*.jsonl.zst
do
    (
    echo ${el}
    zstdcat ${el} | jq -c --arg s ${PERIOD0} --arg e ${PERIOD1} 'select(.ts | . >= $s and . <= $e + "z") | {id: .id, ts: .ts, text: .text, doc_scores: .doc_scores}' | zstd > ${LANG}/${PERIOD0}_${PERIOD1}_$(basename ${el}).jsonl.zst
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

cat ${LANG}/${PERIOD0}_${PERIOD1}_*.zst > ${LANG}/${PERIOD0}_${PERIOD1}.jsonl.zst

rm ${LANG}/${PERIOD0}_${PERIOD1}_*.zst

echo "All done"
