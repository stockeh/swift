#!/bin/bash -l
#PBS -l select=10
#PBS -l walltime=06:00:00
#PBS -l place=scatter
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -A SAFS
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/SAFS/jstock/logs
#PBS -N era5-diffusion

echo "Job started at: $(date '+%Y-%m-%d-%H%M%S')"

module load frameworks
# source /lus/flare/projects/SAFS/jstock/venvs/aur/bin/activate
source /lus/flare/projects/SAFS/jstock/prod/swift/venv/bin/activate

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

# Aurora things
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600 
export PALS_PMI=pmix # Required by Aurora mpich
export CCL_ATL_TRANSPORT=mpi # Required by Aurora mpich

export CCL_OP_SYNC=1
export CCL_ATL_SYNC_COLL=1
export CCL_ENABLE_AUTO_CACHE=0
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=4096

export CCL_ALLREDUCE=topo
export CCL_ALLREDUCE_SCALEOUT=direct
export CCL_ALLGATHERV=direct
export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0

export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_MR_CACHE_MONITOR=disabled
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=30

export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export NUMEXPR_MAX_THREADS=7
export OMP_NUM_THREADS=7

# cd /lus/flare/projects/SAFS/jstock/research/swift
cd /lus/flare/projects/SAFS/jstock/prod/swift
echo "Current working directory: $(pwd)"

source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
ezpz_setup_env

# --------------------------------
# require parameter for experiment
if [[ -z "${EXPERIMENT}" ]]; then
  echo "Error: EXPERIMENT environment variable is not set." >&2
  exit 1
fi

# --------------------------------
# optional local batch size
: ${LOCAL_BATCH_SIZE:=1}

# --------------------------------
# resume setup (using PARTID from environment)
: ${PARTID:=0} # default to 0
if ! [[ "$PARTID" =~ ^[0-9]+$ ]]; then
  echo "Error: PARTID must be a non-negative integer." >&2
  exit 1
fi
PARTID=$(printf "%03d" "$PARTID")
if [[ "$PARTID" == "000" ]]; then
  resume=null
else
  resume=$(printf "%03d" $((10#$PARTID - 1)))
fi
echo "PARTID: $PARTID"

export HYDRA_FULL_ERROR=1
export HYDRA_RUN_ID=$PARTID
BATCH_SIZE=$((num_gpus * LOCAL_BATCH_SIZE))

# finetune=multistep \
run_cmd="${DIST_LAUNCH} python3 -m swift.train \
    experiment=${EXPERIMENT} \
    data.batch_size=${BATCH_SIZE} \
    resume=${resume}"

eval "${run_cmd}"