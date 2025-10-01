#!/bin/bash
#
# job0=$(qsub -v PARTID=0 aurora-unet-1.4.sh)
# job1=$(qsub -W depend=afterany:${job0} -v PARTID=1 aurora-unet-1.4.sh)
#
# bash chain-resume.sh -s 0 -n 4 -b 5 -e era5-swinv2-5.6-scm
#

usage() {
  cat <<EOF
Usage: $0 [-s START] [-n COUNT] [-b BATCH] [-e EXPER] 
  -s START     first PARTID (default 0)
  -n COUNT     number of jobs (default 4)
  -b BATCH     local batch size (default 1)
  -e EXPER     experiment name (required)
EOF
  exit 1
}

SCRIPT="aurora-general.sh"

START=0; COUNT=4; BATCH=1; EXPERIMENT=""
while getopts "s:n:b:e:" opt; do
  case $opt in
    s) START=$OPTARG ;;
    n) COUNT=$OPTARG ;;
    b) BATCH=$OPTARG ;;
    e) EXPERIMENT=$OPTARG ;;
    *) usage ;;
  esac
done

[[ -z $EXPERIMENT ]] && { echo "❌ -e EXPERIMENT is required"; usage; }

prev_job=""

for (( i=START; i<START+COUNT; i++ )); do
  v="PARTID=${i},EXPERIMENT=${EXPERIMENT},LOCAL_BATCH_SIZE=${BATCH}"
  if [[ -z $prev_job ]]; then
    prev_job=$(qsub -v "$v" "$SCRIPT")
  else
    prev_job=$(qsub -W depend=afterany:$prev_job -v "$v" "$SCRIPT")
  fi
  echo "Submitted PARTID=$i → JobID $prev_job"
done