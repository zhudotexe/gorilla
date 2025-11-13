#!/bin/bash
for i in gen/*.sh; do
  if [[ -z "$job_id" ]]; then
    job_id=$(sbatch --parsable "$i");
    echo "$i: $job_id";
  else
    job_id=$(sbatch --parsable --dependency=afterany:$job_id "$i");
    echo "$i: $job_id (DEP)";
  fi
done
