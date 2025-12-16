#! /bin/bash

fields=("ts" "crawl_id")

while read p; 
do
  for f in "${fields[@]}"; 
  do
      sbatch extract.slurm $p $f
  done
done < ${1}

